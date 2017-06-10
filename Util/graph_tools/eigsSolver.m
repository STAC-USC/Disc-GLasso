function [ev, ek] = eigsSolver( L, maxK )

  opts.issym=1;
  opts.isreal=1;
  opts.disp=0;

  [ev, ek] = eigs( L, maxK+1, 'sm', opts );
  
  [ek, idx] = sort( diag(ek) );
  
  ev = ev( :, idx(2:maxK+1) );
  ek = ek(    idx(2:maxK+1) );
end
