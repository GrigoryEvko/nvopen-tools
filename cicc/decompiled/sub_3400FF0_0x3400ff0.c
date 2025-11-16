// Function: sub_3400FF0
// Address: 0x3400ff0
//
unsigned __int8 *__fastcall sub_3400FF0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  int v3; // eax
  _QWORD *v4; // r13
  __int64 v5; // rcx
  __int64 v6; // r8
  _QWORD *v7; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  _DWORD *v12; // r14
  __m128i v13; // xmm0
  unsigned __int16 v14; // cx
  int v15; // eax
  char v16; // r15
  __m128i v17; // [rsp+10h] [rbp-40h] BYREF

  v2 = (__int64 *)*a1;
  v3 = *(unsigned __int16 *)*a1;
  if ( (_WORD)v3 )
  {
    if ( (unsigned __int16)(v3 - 17) <= 0xD3u )
      LOWORD(v3) = word_4456580[v3 - 1];
  }
  else
  {
    if ( !sub_30070B0(*a1) )
    {
      v4 = (_QWORD *)a1[1];
      goto LABEL_10;
    }
    LOWORD(v3) = sub_3009970((__int64)v2, a2, v9, v10, v11);
    v2 = (__int64 *)*a1;
  }
  v4 = (_QWORD *)a1[1];
  if ( (_WORD)v3 == 2 )
    goto LABEL_5;
LABEL_10:
  v12 = (_DWORD *)v4[2];
  v13 = _mm_loadu_si128((const __m128i *)a1[2]);
  v17 = v13;
  if ( !v13.m128i_i16[0] )
  {
    v16 = sub_3007030((__int64)&v17);
    if ( sub_30070B0((__int64)&v17) )
      goto LABEL_21;
    if ( !v16 )
      goto LABEL_14;
LABEL_19:
    v15 = v12[16];
    goto LABEL_15;
  }
  v14 = v13.m128i_i16[0] - 17;
  if ( (unsigned __int16)(v13.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v13.m128i_i16[0] - 126) > 0x31u )
  {
    if ( v14 > 0xD3u )
    {
LABEL_14:
      v15 = v12[15];
      goto LABEL_15;
    }
    goto LABEL_21;
  }
  if ( v14 > 0xD3u )
    goto LABEL_19;
LABEL_21:
  v15 = v12[17];
LABEL_15:
  if ( v15 )
    return sub_3400BD0((__int64)v4, 0, a1[3], *(unsigned int *)v2, v2[1], 0, v13, 0);
LABEL_5:
  v5 = *v2;
  v6 = v2[1];
  v17.m128i_i64[0] = 0;
  v17.m128i_i32[2] = 0;
  v7 = sub_33F17F0(v4, 51, (__int64)&v17, v5, v6);
  if ( v17.m128i_i64[0] )
    sub_B91220((__int64)&v17, v17.m128i_i64[0]);
  return (unsigned __int8 *)v7;
}
