function ev = eigsNorm( ev )
  for i = 1:size(ev, 1)
    ev(i,:) = ev(i,:) / norm(ev(i,:));
  end
end
