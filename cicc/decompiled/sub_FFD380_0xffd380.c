// Function: sub_FFD380
// Address: 0xffd380
//
void __fastcall sub_FFD380(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r12
  const __m128i *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i *v7; // r14
  unsigned __int64 v8; // rax
  __int64 v9; // r15
  int v10; // r13d
  unsigned int v11; // ebx
  __int64 v12; // r12
  unsigned int v13; // r14d
  __m128i *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 *v20; // [rsp+18h] [rbp-338h]
  __int64 v21; // [rsp+20h] [rbp-330h]
  int v22; // [rsp+28h] [rbp-328h]
  __m128i *v23; // [rsp+30h] [rbp-320h]
  __int64 *v24; // [rsp+38h] [rbp-318h]
  __m128i *v25; // [rsp+40h] [rbp-310h] BYREF
  __m128i *v26; // [rsp+48h] [rbp-308h]
  const __m128i *v27; // [rsp+50h] [rbp-300h]
  __m128i v28; // [rsp+60h] [rbp-2F0h] BYREF

  v3 = *(_QWORD *)(a1 + 552);
  if ( !a3 || !v3 )
    return;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v20 = &a2[3 * a3];
  if ( v20 == a2 )
  {
    v15 = 0;
    v14 = 0;
    goto LABEL_25;
  }
  v24 = a2;
  v4 = 0;
  v23 = 0;
  while ( 1 )
  {
    v5 = v24[2];
    v28.m128i_i64[0] = *v24;
    v28.m128i_i64[1] = v5 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v23 != v4 )
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(&v28);
        v4 = v27;
        v23 = v26;
        v5 = v24[2];
      }
      v28.m128i_i64[0] = v5;
      v6 = v24[1];
      v7 = v23 + 1;
      v26 = v23 + 1;
      v28.m128i_i64[1] = v6 & 0xFFFFFFFFFFFFFFFBLL;
      if ( &v23[1] == v4 )
        goto LABEL_41;
LABEL_9:
      *v7 = _mm_loadu_si128(&v28);
      v7 = v26;
      goto LABEL_10;
    }
    sub_F38BA0((const __m128i **)&v25, v23, &v28);
    v7 = v26;
    v19 = v24[1];
    v28.m128i_i64[0] = v24[2];
    v28.m128i_i64[1] = v19 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v27 == v26 )
    {
      v4 = v26;
LABEL_41:
      sub_F38BA0((const __m128i **)&v25, v4, &v28);
      v23 = v26;
      goto LABEL_11;
    }
    if ( v26 )
      goto LABEL_9;
LABEL_10:
    v23 = v7 + 1;
    v26 = v7 + 1;
LABEL_11:
    v21 = *v24;
    v8 = *(_QWORD *)(*v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 == *v24 + 48 )
      goto LABEL_29;
    if ( !v8 )
      BUG();
    v9 = v8 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
      goto LABEL_29;
    v22 = sub_B46E30(v9);
    v10 = v22 >> 2;
    if ( v22 >> 2 > 0 )
    {
      v11 = 0;
      v12 = v24[1];
      while ( v12 != sub_B46EC0(v9, v11) )
      {
        v13 = v11 + 1;
        if ( v12 == sub_B46EC0(v9, v11 + 1)
          || (v13 = v11 + 2, v12 == sub_B46EC0(v9, v11 + 2))
          || (v13 = v11 + 3, v12 == sub_B46EC0(v9, v11 + 3)) )
        {
          if ( v13 != v22 )
            goto LABEL_22;
          goto LABEL_29;
        }
        v11 += 4;
        if ( !--v10 )
        {
          v17 = v22 - v11;
          goto LABEL_35;
        }
      }
LABEL_21:
      if ( v11 != v22 )
        goto LABEL_22;
LABEL_29:
      v16 = v24[1];
      goto LABEL_30;
    }
    v17 = v22;
    v11 = 0;
LABEL_35:
    if ( v17 == 2 )
      goto LABEL_36;
    if ( v17 == 3 )
    {
      if ( v24[1] == sub_B46EC0(v9, v11) )
        goto LABEL_21;
      ++v11;
LABEL_36:
      if ( v24[1] == sub_B46EC0(v9, v11) )
        goto LABEL_21;
      ++v11;
      goto LABEL_38;
    }
    if ( v17 != 1 )
      goto LABEL_29;
LABEL_38:
    v18 = sub_B46EC0(v9, v11);
    v16 = v24[1];
    if ( v16 == v18 )
      goto LABEL_21;
LABEL_30:
    v28.m128i_i64[1] = v16 | 4;
    v28.m128i_i64[0] = v21;
    if ( v27 == v23 )
    {
      sub_F38BA0((const __m128i **)&v25, v23, &v28);
      v23 = v26;
    }
    else
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(&v28);
        v23 = v26;
      }
      v26 = ++v23;
    }
LABEL_22:
    v24 += 3;
    if ( v20 == v24 )
      break;
    v4 = v27;
  }
  v14 = v25;
  v3 = *(_QWORD *)(a1 + 552);
  v15 = v23 - v25;
LABEL_25:
  sub_B26B80((__int64)&v28, (unsigned __int64 *)v14, v15, 1u);
  sub_B2A420(v3, (__int64)&v28, 0);
  sub_B1AA80((__int64)&v28, (__int64)&v28);
  if ( v25 )
    j_j___libc_free_0(v25, (char *)v27 - (char *)v25);
}
