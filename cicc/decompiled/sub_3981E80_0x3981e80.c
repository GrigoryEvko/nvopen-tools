// Function: sub_3981E80
// Address: 0x3981e80
//
__int64 __fastcall sub_3981E80(__int64 a1)
{
  __int16 v1; // ax
  __int16 v2; // ax
  unsigned __int64 v3; // rax

  v1 = *(_WORD *)(a1 + 28);
  if ( v1 == 17 || v1 == 65 )
    return a1;
  do
  {
    v3 = sub_3981CC0(a1);
    a1 = v3;
    if ( !v3 )
      break;
    v2 = *(_WORD *)(v3 + 28);
    if ( v2 == 17 )
      break;
  }
  while ( v2 != 65 );
  return a1;
}
