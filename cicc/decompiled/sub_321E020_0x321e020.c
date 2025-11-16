// Function: sub_321E020
// Address: 0x321e020
//
unsigned __int8 __fastcall sub_321E020(_QWORD *a1, __int32 a2, __int64 a3)
{
  _QWORD *v3; // r14
  __int64 v4; // rax
  _QWORD *v5; // r12
  bool v6; // dl
  _QWORD *v7; // rax
  __int64 v8; // rax
  unsigned __int8 result; // al
  char v10; // r15
  __m128i *v11; // rax
  __m128i v12; // [rsp+10h] [rbp-40h] BYREF

  v3 = a1 + 1;
  v4 = sub_B0D520(a3);
  v12.m128i_i32[0] = a2;
  v5 = (_QWORD *)a1[2];
  v12.m128i_i64[1] = v4;
  if ( !v5 )
  {
    v5 = a1 + 1;
    if ( (_QWORD *)a1[3] == v3 )
    {
      v10 = 1;
      goto LABEL_11;
    }
LABEL_9:
    v8 = sub_220EF80((__int64)v5);
    result = sub_321DFB0(v8 + 32, (__int64)&v12);
    if ( !result )
      return result;
LABEL_10:
    v10 = 1;
    if ( v3 == v5 )
    {
LABEL_11:
      v11 = (__m128i *)sub_22077B0(0x30u);
      v11[2] = _mm_loadu_si128(&v12);
      result = (unsigned __int8)sub_220F040(v10, (__int64)v11, v5, a1 + 1);
      ++a1[5];
      return result;
    }
LABEL_15:
    v10 = sub_321DFB0((__int64)&v12, (__int64)(v5 + 4));
    goto LABEL_11;
  }
  while ( 1 )
  {
    v6 = sub_321DFB0((__int64)&v12, (__int64)(v5 + 4));
    v7 = (_QWORD *)v5[3];
    if ( v6 )
      v7 = (_QWORD *)v5[2];
    if ( !v7 )
      break;
    v5 = v7;
  }
  if ( v6 )
  {
    if ( (_QWORD *)a1[3] == v5 )
      goto LABEL_10;
    goto LABEL_9;
  }
  result = sub_321DFB0((__int64)(v5 + 4), (__int64)&v12);
  if ( result )
  {
    v10 = 1;
    if ( v3 == v5 )
      goto LABEL_11;
    goto LABEL_15;
  }
  return result;
}
