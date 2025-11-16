// Function: sub_C34710
// Address: 0xc34710
//
__int64 __fastcall sub_C34710(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        char a6,
        _BYTE *a7)
{
  unsigned int v10; // r12d
  unsigned int v12; // esi
  unsigned int v13; // edx
  int v14; // [rsp+14h] [rbp-34h]

  v14 = a5;
  v10 = sub_C34470(a1, a2, a3, a4, a5, a6, a7);
  if ( v10 == 1 )
  {
    v12 = (a4 + 63) >> 6;
    if ( !v12 )
      v12 = 1;
    v13 = 0;
    if ( (*(_BYTE *)(a1 + 20) & 7) != 1 )
    {
      v13 = a4 - v14;
      if ( (*(_BYTE *)(a1 + 20) & 8) != 0 )
        v13 = v14;
    }
    sub_C31E80((char *)a2, v12, v13);
    if ( (*(_BYTE *)(a1 + 20) & 8) != 0 && a5 )
      sub_C475D0(a2);
  }
  return v10;
}
