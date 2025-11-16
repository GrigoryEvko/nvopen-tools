// Function: sub_150F0A0
// Address: 0x150f0a0
//
__int64 __fastcall sub_150F0A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i *a8,
        unsigned __int64 a9)
{
  char v9; // dl
  char v10; // al
  __int64 v11; // rax
  __int64 v13; // rdx
  __m128i v14[4]; // [rsp+0h] [rbp-60h] BYREF
  char v15; // [rsp+40h] [rbp-20h]

  sub_14F5920(v14, a2, a3, a4, a5, a6, a7, a8, a9);
  v9 = v15 & 1;
  v10 = (2 * (v15 & 1)) | v15 & 0xFD;
  v15 = v10;
  if ( v9 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v15 = v10 & 0xFD;
    v11 = v14[0].m128i_i64[0];
    v14[0].m128i_i64[0] = 0;
    *(_QWORD *)a1 = v11 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    sub_150E2B0(a1, (__int64)v14);
    if ( (v15 & 2) != 0 )
      sub_14F5A70(v14, (__int64)v14, v13);
    if ( (v15 & 1) == 0 )
      return a1;
  }
  if ( v14[0].m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14[0].m128i_i64[0] + 8LL))(v14[0].m128i_i64[0]);
  return a1;
}
