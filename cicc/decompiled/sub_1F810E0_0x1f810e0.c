// Function: sub_1F810E0
// Address: 0x1f810e0
//
__int64 *__fastcall sub_1F810E0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        const void **a4,
        unsigned __int64 a5,
        __int16 *a6,
        __m128 a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v17; // rax
  char v18; // dl
  __int64 v19; // rax
  unsigned int v20; // esi
  bool v22; // al
  unsigned __int64 v23; // [rsp+8h] [rbp-48h]
  _BYTE v24[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h]

  v17 = *(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6;
  v18 = *(_BYTE *)v17;
  v19 = *(_QWORD *)(v17 + 8);
  v24[0] = v18;
  v25 = v19;
  if ( v18 )
  {
    v20 = ((unsigned __int8)(v18 - 14) < 0x60u) + 134;
  }
  else
  {
    v23 = a3;
    v22 = sub_1F58D20((__int64)v24);
    a3 = v23;
    v20 = 134 - (!v22 - 1);
  }
  return sub_1D3A900(a1, v20, a2, a3, a4, 0, a7, a8, a9, a5, a6, a10, a11, a12);
}
