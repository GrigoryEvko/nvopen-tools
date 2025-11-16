// Function: sub_9F1E00
// Address: 0x9f1e00
//
__int64 __fastcall sub_9F1E00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        const __m128i *a8,
        unsigned __int64 a9)
{
  __int64 v9; // rdx
  char v10; // al
  __int64 v11; // rax
  __m128i v13[4]; // [rsp+0h] [rbp-60h] BYREF
  char v14; // [rsp+40h] [rbp-20h]

  sub_9D5100(v13, a2, a3, a4, a5, a6, a7, a8, a9);
  v9 = v14 & 1;
  v10 = (2 * v9) | v14 & 0xFD;
  v14 = v10;
  if ( (_BYTE)v9 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v14 = v10 & 0xFD;
    v11 = v13[0].m128i_i64[0];
    v13[0].m128i_i64[0] = 0;
    *(_QWORD *)a1 = v11 & 0xFFFFFFFFFFFFFFFELL;
  }
  else
  {
    sub_9F0EA0(a1, (__int64)v13, v9, (unsigned int)(2 * v9));
    if ( (v14 & 2) != 0 )
      sub_9D52C0(v13);
    if ( (v14 & 1) == 0 )
      return a1;
  }
  if ( v13[0].m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13[0].m128i_i64[0] + 8LL))(v13[0].m128i_i64[0]);
  return a1;
}
