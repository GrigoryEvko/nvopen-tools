// Function: sub_370D170
// Address: 0x370d170
//
__int64 *__fastcall sub_370D170(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  __m128i v7[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v8; // [rsp+30h] [rbp-20h]

  if ( a2[9] && !a2[7] && !a2[8] )
  {
    sub_37074E0(v7[0].m128i_i64, *(_QWORD *)(a4 + 8), *(_QWORD *)(a4 + 16), (__int64)a2);
    v5 = v7[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v7[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v7[0].m128i_i64[0] = 0;
      sub_9C66B0(v7[0].m128i_i64);
      goto LABEL_9;
    }
    *a1 = 0;
    v7[0].m128i_i64[0] = v5 | 1;
    sub_9C6670(a1, v7);
    sub_9C66B0(v7[0].m128i_i64);
    return a1;
  }
  v8 = 257;
  sub_3701060(&v6, a2 + 2, (_QWORD *)(a4 + 8), v7);
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v6 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
LABEL_9:
  *a1 = 1;
  return a1;
}
