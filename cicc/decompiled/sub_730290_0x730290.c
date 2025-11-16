// Function: sub_730290
// Address: 0x730290
//
__int64 __fastcall sub_730290(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  char v3; // cl

  v1 = sub_730250(a1);
  if ( !v1 )
    return a1;
  v2 = *(_QWORD *)(v1 + 144);
  if ( !v2 )
    return a1;
  v3 = *(_BYTE *)(v2 + 24);
  if ( v3 == 5 )
  {
    if ( (*(_BYTE *)(v1 + 171) & 2) == 0 )
      return a1;
  }
  else if ( v3 != 31 )
  {
    return a1;
  }
  return *(_QWORD *)(v2 + 56);
}
