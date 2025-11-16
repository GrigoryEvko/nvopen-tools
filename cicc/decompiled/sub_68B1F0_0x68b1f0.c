// Function: sub_68B1F0
// Address: 0x68b1f0
//
_BOOL8 __fastcall sub_68B1F0(__int64 *a1, __int64 *a2, _DWORD *a3)
{
  char v3; // al
  __int64 v5; // r12

  *a3 = 0;
  if ( (*(_BYTE *)(unk_4D03C50 + 17LL) & 1) == 0 )
    return 0;
  v3 = *((_BYTE *)a2 + 16);
  *a3 = v3 == 2;
  if ( *((_BYTE *)a1 + 16) == 2 )
  {
    if ( v3 == 2 )
      return 0;
    v5 = *a2;
  }
  else
  {
    if ( v3 != 2 )
      return 0;
    v5 = *a1;
  }
  if ( !v5 || !(unsigned int)sub_8D2930(v5) )
    return 0;
  return (unsigned int)sub_8D27E0(v5) == 0;
}
