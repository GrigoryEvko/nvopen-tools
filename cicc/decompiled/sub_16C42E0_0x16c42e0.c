// Function: sub_16C42E0
// Address: 0x16c42e0
//
__int64 __fastcall sub_16C42E0(unsigned __int8 *a1, unsigned __int64 a2, int a3)
{
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  char *v6; // rax
  _QWORD *v7; // rax
  bool v8; // bl
  unsigned __int8 v10; // dl
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-C0h] BYREF
  __m128i v14; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v15; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v16; // [rsp+30h] [rbp-90h] BYREF
  __m128i v17; // [rsp+40h] [rbp-80h]
  __m128i v18; // [rsp+50h] [rbp-70h]
  _QWORD v19[12]; // [rsp+60h] [rbp-60h] BYREF

  sub_16C36E0((__int64)&v13, a1, a2, a3);
  v4 = _mm_loadu_si128(&v14);
  v5 = _mm_loadu_si128(&v15);
  v16 = _mm_loadu_si128(&v13);
  v17 = v4;
  v18 = v5;
  sub_16C3680(v19, (__int64)a1, a2);
  if ( sub_16C36A0(&v13, v19) )
    return 0;
  if ( v14.m128i_i64[1] > 2uLL && sub_16C36C0(*(_BYTE *)v14.m128i_i64[0], a3) )
  {
    v6 = (char *)v14.m128i_i64[0];
    v10 = *(_BYTE *)(v14.m128i_i64[0] + 1);
    v8 = *(_BYTE *)v14.m128i_i64[0] == v10;
    if ( a3 || !v14.m128i_i64[1] || *(_BYTE *)(v14.m128i_i64[0] + v14.m128i_i64[1] - 1) != 58 )
    {
      if ( *(_BYTE *)v14.m128i_i64[0] == v10 )
      {
        v11 = (_QWORD *)sub_16C3860((__int64)&v16);
        if ( !sub_16C36A0(v11, v19) && sub_16C36C0(*(_BYTE *)v17.m128i_i64[0], a3) )
          return v17.m128i_i64[0];
        return 0;
      }
LABEL_11:
      if ( sub_16C36C0(*v6, a3) )
        return v14.m128i_i64[0];
      return 0;
    }
    v12 = (_QWORD *)sub_16C3860((__int64)&v16);
    if ( sub_16C36A0(v12, v19) )
      goto LABEL_8;
  }
  else
  {
    if ( a3 || !v14.m128i_i64[1] )
      goto LABEL_10;
    v6 = (char *)v14.m128i_i64[0];
    if ( *(_BYTE *)(v14.m128i_i64[0] + v14.m128i_i64[1] - 1) != 58 )
      goto LABEL_11;
    v7 = (_QWORD *)sub_16C3860((__int64)&v16);
    v8 = sub_16C36A0(v7, v19);
    if ( v8 )
      goto LABEL_10;
  }
  if ( !sub_16C36C0(*(_BYTE *)v17.m128i_i64[0], 0) )
  {
LABEL_8:
    if ( v8 )
      return 0;
LABEL_10:
    v6 = (char *)v14.m128i_i64[0];
    goto LABEL_11;
  }
  return v17.m128i_i64[0];
}
