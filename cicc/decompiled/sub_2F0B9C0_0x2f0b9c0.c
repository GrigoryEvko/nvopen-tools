// Function: sub_2F0B9C0
// Address: 0x2f0b9c0
//
void __fastcall sub_2F0B9C0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  unsigned int v4; // eax
  const __m128i *v5; // r14
  const __m128i *v6; // rax
  __int64 v7; // rax
  const __m128i *v8; // r15
  const __m128i *v9; // rbx
  __int64 v10; // rax
  __m128i *i; // r13
  const __m128i *v12; // rbx
  __m128i *j; // r15
  bool v14; // zf
  unsigned __int64 *k; // rbx
  unsigned __int64 *m; // r15
  unsigned __int64 v17; // r15
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r14
  const __m128i *v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned int v27; // [rsp+18h] [rbp-68h]
  unsigned int v28; // [rsp+1Ch] [rbp-64h]
  unsigned __int64 v29; // [rsp+28h] [rbp-58h]
  unsigned int v30; // [rsp+30h] [rbp-50h]
  unsigned int v31; // [rsp+34h] [rbp-4Ch]
  bool v32; // [rsp+34h] [rbp-4Ch]
  const __m128i *v33; // [rsp+38h] [rbp-48h]
  unsigned __int64 v34; // [rsp+40h] [rbp-40h]
  unsigned __int64 v35; // [rsp+48h] [rbp-38h]
  __int64 v36; // [rsp+48h] [rbp-38h]

  v3 = a1;
  v4 = *(_DWORD *)a1;
  v5 = *(const __m128i **)(a1 + 16);
  *(_QWORD *)(a1 + 16) = 0;
  v30 = v4;
  v25 = v5;
  v27 = *(_DWORD *)(a1 + 4);
  v6 = *(const __m128i **)(a1 + 8);
  *(_QWORD *)(a1 + 8) = 0;
  v33 = v6;
  v7 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = 0;
  v26 = v7;
  while ( 1 )
  {
    v8 = *(const __m128i **)(v3 - 16);
    v9 = *(const __m128i **)(v3 - 24);
    v31 = *(_DWORD *)(v3 - 32);
    v28 = *(_DWORD *)(v3 - 28);
    v29 = (char *)v8 - (char *)v9;
    if ( v8 == v9 )
    {
      v35 = 0;
    }
    else
    {
      if ( (unsigned __int64)((char *)v8 - (char *)v9) > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_47;
      a1 = (char *)v8 - (char *)v9;
      v10 = sub_22077B0(v29);
      v8 = *(const __m128i **)(v3 - 16);
      v9 = *(const __m128i **)(v3 - 24);
      v35 = v10;
    }
    for ( i = (__m128i *)v35; v8 != v9; i = (__m128i *)((char *)i + 56) )
    {
      if ( i )
      {
        a1 = (unsigned __int64)i;
        i->m128i_i64[0] = (__int64)i[1].m128i_i64;
        a2 = v9->m128i_i64[0];
        sub_2F07250(i->m128i_i64, v9->m128i_i64[0], v9->m128i_i64[0] + v9->m128i_i64[1]);
        i[2] = _mm_loadu_si128(v9 + 2);
        i[3].m128i_i16[0] = v9[3].m128i_i16[0];
      }
      v9 = (const __m128i *)((char *)v9 + 56);
    }
    if ( v5 == v33 )
    {
      v34 = 0;
    }
    else
    {
      if ( (unsigned __int64)((char *)v5 - (char *)v33) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_47:
        sub_4261EA(a1, a2, a3);
      a1 = (char *)v5 - (char *)v33;
      v34 = sub_22077B0((char *)v5 - (char *)v33);
    }
    v12 = v33;
    for ( j = (__m128i *)v34; v5 != v12; j = (__m128i *)((char *)j + 56) )
    {
      if ( j )
      {
        a1 = (unsigned __int64)j;
        j->m128i_i64[0] = (__int64)j[1].m128i_i64;
        a2 = v12->m128i_i64[0];
        sub_2F07250(j->m128i_i64, v12->m128i_i64[0], v12->m128i_i64[0] + v12->m128i_i64[1]);
        j[2] = _mm_loadu_si128(v12 + 2);
        j[3].m128i_i16[0] = v12[3].m128i_i16[0];
      }
      v12 = (const __m128i *)((char *)v12 + 56);
    }
    v14 = v31 == v30;
    v32 = v31 > v30;
    if ( v14 )
      v32 = v27 < v28;
    for ( k = (unsigned __int64 *)v34; k != (unsigned __int64 *)j; k += 7 )
    {
      a1 = *k;
      if ( (unsigned __int64 *)*k != k + 2 )
      {
        a2 = k[2] + 1;
        j_j___libc_free_0(a1);
      }
    }
    if ( v34 )
    {
      a2 = (char *)v5 - (char *)v33;
      a1 = v34;
      j_j___libc_free_0(v34);
    }
    for ( m = (unsigned __int64 *)v35; m != (unsigned __int64 *)i; m += 7 )
    {
      a1 = *m;
      if ( (unsigned __int64 *)*m != m + 2 )
      {
        a2 = m[2] + 1;
        j_j___libc_free_0(a1);
      }
    }
    if ( v35 )
    {
      a2 = v29;
      a1 = v35;
      j_j___libc_free_0(v35);
    }
    v17 = *(_QWORD *)(v3 + 8);
    v18 = *(unsigned __int64 **)(v3 + 16);
    v36 = *(_QWORD *)(v3 + 24);
    if ( !v32 )
      break;
    v19 = *(unsigned __int64 **)(v3 + 8);
    *(_QWORD *)v3 = *(_QWORD *)(v3 - 32);
    v20 = *(_QWORD *)(v3 - 24);
    *(_QWORD *)(v3 - 24) = 0;
    *(_QWORD *)(v3 + 8) = v20;
    v21 = *(_QWORD *)(v3 - 16);
    *(_QWORD *)(v3 - 16) = 0;
    *(_QWORD *)(v3 + 16) = v21;
    v22 = *(_QWORD *)(v3 - 8);
    *(_QWORD *)(v3 - 8) = 0;
    *(_QWORD *)(v3 + 24) = v22;
    if ( (unsigned __int64 *)v17 != v18 )
    {
      do
      {
        a1 = *v19;
        if ( (unsigned __int64 *)*v19 != v19 + 2 )
        {
          a2 = v19[2] + 1;
          j_j___libc_free_0(a1);
        }
        v19 += 7;
      }
      while ( v19 != v18 );
    }
    if ( v17 )
    {
      a1 = v17;
      a2 = v36 - v17;
      j_j___libc_free_0(v17);
    }
    v3 -= 32LL;
  }
  v23 = *(unsigned __int64 **)(v3 + 16);
  v24 = *(unsigned __int64 **)(v3 + 8);
  *(_DWORD *)v3 = v30;
  *(_DWORD *)(v3 + 4) = v27;
  *(_QWORD *)(v3 + 8) = v33;
  *(_QWORD *)(v3 + 16) = v25;
  *(_QWORD *)(v3 + 24) = v26;
  if ( (unsigned __int64 *)v17 != v23 )
  {
    do
    {
      if ( (unsigned __int64 *)*v24 != v24 + 2 )
        j_j___libc_free_0(*v24);
      v24 += 7;
    }
    while ( v24 != v23 );
  }
  if ( v17 )
    j_j___libc_free_0(v17);
}
