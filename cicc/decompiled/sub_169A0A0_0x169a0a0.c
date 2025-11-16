// Function: sub_169A0A0
// Address: 0x169a0a0
//
__int64 __fastcall sub_169A0A0(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        unsigned int a6,
        _BYTE *a7)
{
  unsigned int v9; // r12d
  __int64 v11; // rdx
  unsigned int v12; // [rsp+14h] [rbp-34h]

  v12 = a5;
  v9 = sub_1699DC0(a1, a2, a3, a4, a5, a6, a7);
  if ( v9 == 1 )
  {
    v11 = 0;
    if ( (*(_BYTE *)(a1 + 18) & 7) != 1 )
    {
      v11 = a4 - v12;
      if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
        v11 = v12;
    }
    sub_16AEAC0(a2, (a4 + 63) >> 6, v11);
    if ( (*(_BYTE *)(a1 + 18) & 8) != 0 && a5 )
      sub_16A7D00(a2);
  }
  return v9;
}
