// Function: sub_2E6B8A0
// Address: 0x2e6b8a0
//
void __fastcall sub_2E6B8A0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r12
  _QWORD *v4; // r13
  _QWORD *v5; // r15
  const __m128i *v6; // r8
  __m128i *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i *v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  _QWORD *v17; // rcx
  unsigned __int64 *v18; // r8
  __int64 v19; // rdx
  unsigned __int64 *v20; // [rsp+0h] [rbp-310h] BYREF
  __m128i *v21; // [rsp+8h] [rbp-308h]
  const __m128i *v22; // [rsp+10h] [rbp-300h]
  __m128i v23; // [rsp+20h] [rbp-2F0h] BYREF

  v3 = *(_QWORD *)(a1 + 552);
  if ( !a3 || !v3 )
    return;
  v4 = a2;
  v20 = 0;
  v21 = 0;
  v5 = &a2[3 * a3];
  v22 = 0;
  if ( v5 == a2 )
  {
    v19 = 0;
    v18 = 0;
    goto LABEL_22;
  }
  v6 = 0;
  v7 = 0;
  while ( 1 )
  {
    v8 = v4[2];
    v23.m128i_i64[0] = *v4;
    v23.m128i_i64[1] = v8 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v7 == v6 )
    {
      sub_2E649C0((unsigned __int64 *)&v20, v7, &v23);
      v11 = v4[1];
      v10 = v21;
      v23.m128i_i64[0] = v4[2];
      v23.m128i_i64[1] = v11 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v22 != v21 )
      {
        if ( v21 )
        {
LABEL_9:
          *v10 = _mm_loadu_si128(&v23);
          v10 = v21;
          v11 = v4[1];
        }
        v7 = v10 + 1;
        v21 = v7;
        goto LABEL_11;
      }
      v6 = v21;
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128(&v23);
        v6 = v22;
        v7 = v21;
        v8 = v4[2];
      }
      v23.m128i_i64[0] = v8;
      v9 = v4[1];
      v10 = v7 + 1;
      v21 = v10;
      v23.m128i_i64[1] = v9 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v10 != v6 )
        goto LABEL_9;
    }
    sub_2E649C0((unsigned __int64 *)&v20, v6, &v23);
    v7 = v21;
    v11 = v4[1];
LABEL_11:
    v12 = *(_QWORD **)(*v4 + 112LL);
    v13 = 8LL * *(unsigned int *)(*v4 + 120LL);
    v14 = &v12[(unsigned __int64)v13 / 8];
    v15 = v13 >> 3;
    v16 = v13 >> 5;
    if ( v16 )
    {
      v17 = &v12[4 * v16];
      while ( *v12 != v11 )
      {
        if ( v12[1] == v11 )
        {
          ++v12;
          break;
        }
        if ( v12[2] == v11 )
        {
          v12 += 2;
          break;
        }
        if ( v12[3] == v11 )
        {
          v12 += 3;
          break;
        }
        v12 += 4;
        if ( v17 == v12 )
        {
          v15 = v14 - v12;
          goto LABEL_26;
        }
      }
LABEL_18:
      if ( v14 != v12 )
        goto LABEL_19;
      goto LABEL_30;
    }
LABEL_26:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_30;
        goto LABEL_29;
      }
      if ( *v12 == v11 )
        goto LABEL_18;
      ++v12;
    }
    if ( *v12 == v11 )
      goto LABEL_18;
    ++v12;
LABEL_29:
    if ( *v12 == v11 )
      goto LABEL_18;
LABEL_30:
    v23.m128i_i64[0] = *v4;
    v23.m128i_i64[1] = v11 | 4;
    if ( v22 == v7 )
    {
      sub_2E649C0((unsigned __int64 *)&v20, v7, &v23);
      v7 = v21;
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128(&v23);
        v7 = v21;
      }
      v21 = ++v7;
    }
LABEL_19:
    v4 += 3;
    if ( v5 == v4 )
      break;
    v6 = v22;
  }
  v18 = v20;
  v3 = *(_QWORD *)(a1 + 552);
  v19 = ((char *)v7 - (char *)v20) >> 4;
LABEL_22:
  sub_2E6AFF0((__int64)&v23, v18, v19, 1u);
  sub_2EBBEA0(v3, &v23, 0);
  sub_2E647F0((__int64)&v23);
  if ( v20 )
    j_j___libc_free_0((unsigned __int64)v20);
}
