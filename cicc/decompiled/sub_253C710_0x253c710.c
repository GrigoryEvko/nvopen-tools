// Function: sub_253C710
// Address: 0x253c710
//
__int64 *__fastcall sub_253C710(__int64 *a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a2 + 97);
  if ( (v2 & 3) == 3 )
  {
    sub_253C590(a1, "readnone");
    return a1;
  }
  else if ( (v2 & 2) != 0 )
  {
    sub_253C590(a1, "readonly");
    return a1;
  }
  else
  {
    if ( (v2 & 1) != 0 )
      sub_253C590(a1, "writeonly");
    else
      sub_253C590(a1, "may-read/write");
    return a1;
  }
}
