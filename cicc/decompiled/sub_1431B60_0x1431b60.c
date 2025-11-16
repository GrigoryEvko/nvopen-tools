// Function: sub_1431B60
// Address: 0x1431b60
//
char __fastcall sub_1431B60(__int64 a1, __m128i *a2, __m128i *a3)
{
  __m128i *v3; // rbx
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __m128i *v7; // rcx
  __m128i *v8; // r12
  void *v9; // r13
  size_t v10; // r8
  size_t v11; // r10
  unsigned __int64 v12; // r14
  __int64 v13; // r15
  void *v14; // rax
  bool v15; // r12
  __m128i *v16; // rax
  unsigned __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rax
  _DWORD *v21; // [rsp+0h] [rbp-70h]
  size_t v22; // [rsp+8h] [rbp-68h]
  __m128i *v23; // [rsp+10h] [rbp-60h]
  size_t v24; // [rsp+10h] [rbp-60h]
  size_t v25; // [rsp+18h] [rbp-58h]
  size_t v26; // [rsp+18h] [rbp-58h]
  size_t v27; // [rsp+18h] [rbp-58h]
  __m128i *v29; // [rsp+28h] [rbp-48h]
  __m128i *v30; // [rsp+38h] [rbp-38h] BYREF

  v3 = a2;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD **)(a1 + 8);
  v21 = (_DWORD *)a1;
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &v5[5 * v4]; i != v5; v5 += 5 )
  {
    if ( v5 )
    {
      *v5 = 0;
      v5[1] = -1;
      v5[2] = 0;
      v5[3] = 0;
      v5[4] = 0;
    }
  }
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v7 = (__m128i *)v3[1].m128i_i64[1];
      v8 = (__m128i *)v3[1].m128i_i64[0];
      v17 = v3->m128i_i64[0];
      v18 = v3->m128i_i64[1];
      v10 = (char *)v7 - (char *)v8;
      v29 = (__m128i *)((char *)v7 - (char *)v8);
      if ( v7 == v8 )
      {
        v11 = 0;
        v9 = 0;
      }
      else
      {
        if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_39;
        a1 = v3[1].m128i_i64[1] - (_QWORD)v8;
        v5 = (_QWORD *)sub_22077B0(v10);
        v7 = (__m128i *)v3[1].m128i_i64[1];
        v8 = (__m128i *)v3[1].m128i_i64[0];
        v9 = v5;
        v10 = (char *)v7 - (char *)v8;
        v11 = (char *)v7 - (char *)v8;
      }
      if ( v7 != v8 )
      {
        a2 = v8;
        a1 = (__int64)v9;
        v22 = v10;
        v23 = v7;
        v25 = v11;
        LOBYTE(v5) = (unsigned __int8)memmove(v9, v8, v11);
        v10 = v22;
        v7 = v23;
        v11 = v25;
      }
      if ( v18 != -1 || v17 )
        break;
      if ( v10 )
      {
        v12 = v3->m128i_i64[0];
        v13 = v3->m128i_i64[1];
LABEL_31:
        if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_39:
          sub_4261EA(a1, a2, i);
        v27 = v11;
        v19 = sub_22077B0(v11);
        v8 = (__m128i *)v3[1].m128i_i64[0];
        v11 = v27;
        a1 = v19;
        v10 = v3[1].m128i_i64[1] - (_QWORD)v8;
        if ( v8 == (__m128i *)v3[1].m128i_i64[1] )
        {
          LOBYTE(v5) = v13 == -2;
          v15 = !((unsigned __int8)v5 & ((v10 | v12) == 0));
          if ( a1 )
          {
LABEL_16:
            a2 = (__m128i *)v11;
            LOBYTE(v5) = j_j___libc_free_0(a1, v11);
          }
LABEL_17:
          if ( v9 )
          {
            a2 = v29;
            a1 = (__int64)v9;
            LOBYTE(v5) = j_j___libc_free_0(v9, v29);
          }
          if ( v15 )
          {
            sub_142F320(v21, (const void **)v3, (size_t **)&v30);
            v16 = v30;
            *v30 = _mm_loadu_si128(v3);
            a1 = v16[1].m128i_i64[0];
            a2 = (__m128i *)v16[2].m128i_i64[0];
            v16[1].m128i_i64[0] = v3[1].m128i_i64[0];
            v16[1].m128i_i64[1] = v3[1].m128i_i64[1];
            i = (_QWORD *)v3[2].m128i_i64[0];
            v16[2].m128i_i64[0] = (__int64)i;
            v3[1].m128i_i64[0] = 0;
            v3[1].m128i_i64[1] = 0;
            v3[2].m128i_i64[0] = 0;
            if ( a1 )
            {
              a2 = (__m128i *)((char *)a2 - a1);
              j_j___libc_free_0(a1, a2);
            }
            LOBYTE(v5) = (_BYTE)v21;
            ++v21[4];
          }
          goto LABEL_23;
        }
LABEL_15:
        v24 = v11;
        v26 = v10;
        v14 = memmove((void *)a1, v8, v10);
        v11 = v24;
        a1 = (__int64)v14;
        v15 = v13 != -2 || (v26 | v12) != 0;
        goto LABEL_16;
      }
      if ( !v9 )
        goto LABEL_24;
      a2 = v29;
      a1 = (__int64)v9;
      LOBYTE(v5) = j_j___libc_free_0(v9, v29);
LABEL_23:
      v8 = (__m128i *)v3[1].m128i_i64[0];
LABEL_24:
      if ( v8 )
      {
        a1 = (__int64)v8;
        a2 = (__m128i *)(v3[2].m128i_i64[0] - (_QWORD)v8);
        LOBYTE(v5) = j_j___libc_free_0(v8, a2);
      }
      v3 = (__m128i *)((char *)v3 + 40);
      if ( a3 == v3 )
        return (char)v5;
    }
    v12 = v3->m128i_i64[0];
    v13 = v3->m128i_i64[1];
    if ( v10 )
      goto LABEL_31;
    a1 = 0;
    if ( v7 == v8 )
    {
      LOBYTE(v5) = v13 == -2;
      v15 = !((unsigned __int8)v5 & (v12 == 0));
      goto LABEL_17;
    }
    goto LABEL_15;
  }
  return (char)v5;
}
