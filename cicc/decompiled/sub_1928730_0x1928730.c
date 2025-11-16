// Function: sub_1928730
// Address: 0x1928730
//
__int64 __fastcall sub_1928730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // r15
  __int64 v7; // r14
  int *v8; // rbx
  bool v9; // zf
  int *v10; // rax
  __int64 v11; // r13
  __m128i v12; // xmm0
  __int64 v13; // r14
  __int64 v14; // r12
  int *v15; // r13
  char v16; // r9
  int *v17; // rax
  __int64 result; // rax
  int *v19; // rax
  __int64 v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+28h] [rbp-78h]
  __m128i v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+58h] [rbp-48h] BYREF
  __m128i v26[4]; // [rsp+60h] [rbp-40h] BYREF

  v24.m128i_i64[0] = a5;
  v24.m128i_i64[1] = a6;
  v23 = (a3 - 1) / 2;
  v22 = a3 & 1;
  if ( a2 >= v23 )
  {
    v8 = (int *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
    {
      v25 = a4;
      goto LABEL_14;
    }
    v11 = a2;
    goto LABEL_17;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = (int *)(a1 + 16 * (i + 1));
    v9 = (unsigned __int8)sub_1921830(v24.m128i_i64, v8, (unsigned int *)(a1 + 8 * (v7 - 1))) == 0;
    v10 = (int *)(a1 + 8 * i);
    if ( !v9 )
      v8 = (int *)(a1 + 8 * --v7);
    *v10 = *v8;
    v10[1] = v8[1];
    if ( v7 >= v23 )
      break;
  }
  v11 = v7;
  if ( !v22 )
  {
LABEL_17:
    if ( (a3 - 2) / 2 == v11 )
    {
      v11 = 2 * v11 + 1;
      v19 = (int *)(a1 + 8 * v11);
      *v8 = *v19;
      v8[1] = v19[1];
      v8 = v19;
    }
  }
  v12 = _mm_loadu_si128(&v24);
  v25 = a4;
  v26[0] = v12;
  v13 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    v14 = v11;
    while ( 1 )
    {
      v15 = (int *)(a1 + 8 * v13);
      v16 = sub_1921830(v26[0].m128i_i64, v15, (unsigned int *)&v25);
      v17 = (int *)(a1 + 8 * v14);
      if ( !v16 )
      {
        v8 = (int *)(a1 + 8 * v14);
        goto LABEL_14;
      }
      v14 = v13;
      *v17 = *v15;
      v17[1] = v15[1];
      if ( a2 >= v13 )
        break;
      v13 = (v13 - 1) / 2;
    }
    v8 = (int *)(a1 + 8 * v13);
  }
LABEL_14:
  result = v25;
  *(_QWORD *)v8 = v25;
  return result;
}
