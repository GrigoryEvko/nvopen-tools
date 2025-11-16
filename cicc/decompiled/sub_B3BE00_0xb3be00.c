// Function: sub_B3BE00
// Address: 0xb3be00
//
void __fastcall sub_B3BE00(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // r15
  unsigned __int64 v6; // rbx
  _QWORD *v7; // r8
  unsigned __int64 v8; // r9
  const __m128i *v9; // rsi
  __m128i *v10; // rdx
  const __m128i *v11; // rax
  const __m128i *v12; // r13
  __m128i *v13; // rdi
  const __m128i *v14; // rsi
  _QWORD *v15; // r13
  _QWORD *v16; // rbx
  _QWORD *v17; // rbx
  _QWORD *v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r13
  _QWORD *v21; // rbx
  __int64 *v22; // r15
  _BYTE *v23; // r13
  __int64 *v24; // r8
  __int64 v25; // rdx
  _BYTE *v26; // rax
  __int64 *v27; // rdi
  size_t v28; // rdx
  _QWORD *v29; // rbx
  __int64 v30; // r9
  __int64 *v31; // r13
  _BYTE *v32; // rbx
  __int64 *v33; // r10
  __int64 v34; // rdx
  _BYTE *v35; // rax
  __int64 *v36; // rdi
  size_t v37; // rdx
  __int64 *v38; // [rsp-50h] [rbp-50h]
  int v39; // [rsp-44h] [rbp-44h]
  _QWORD *v40; // [rsp-40h] [rbp-40h]
  __int64 *v41; // [rsp-40h] [rbp-40h]
  unsigned __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v3 = a2 + 16;
  v4 = a2;
  v5 = *(_QWORD **)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD **)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v17 = &v5[4 * v6];
    if ( v17 != v5 )
    {
      do
      {
        v17 -= 4;
        if ( (_QWORD *)*v17 != v17 + 2 )
        {
          a2 = v17[2] + 1LL;
          j_j___libc_free_0(*v17, a2);
        }
      }
      while ( v17 != v5 );
      v7 = *(_QWORD **)a1;
    }
    if ( v7 != (_QWORD *)(a1 + 16) )
      _libc_free(v7, a2);
    *(_QWORD *)a1 = *(_QWORD *)v4;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(v4 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(v4 + 12);
    *(_QWORD *)v4 = v3;
    *(_QWORD *)(v4 + 8) = 0;
    return;
  }
  v8 = *(unsigned int *)(a2 + 8);
  v39 = *(_DWORD *)(a2 + 8);
  if ( v8 <= v6 )
  {
    v18 = *(_QWORD **)a1;
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_28:
      v19 = &v18[4 * v6];
      while ( v7 != v19 )
      {
        v19 -= 4;
        if ( (_QWORD *)*v19 != v19 + 2 )
        {
          v40 = v7;
          j_j___libc_free_0(*v19, v19[2] + 1LL);
          v7 = v40;
        }
      }
      *(_DWORD *)(a1 + 8) = v39;
      v20 = *(_QWORD **)a2;
      v21 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
      if ( *(_QWORD **)a2 != v21 )
      {
        do
        {
          v21 -= 4;
          if ( (_QWORD *)*v21 != v21 + 2 )
            j_j___libc_free_0(*v21, v21[2] + 1LL);
        }
        while ( v20 != v21 );
      }
      goto LABEL_18;
    }
    v30 = 32 * v8;
    v31 = v5 + 2;
    v32 = (_BYTE *)(a2 + 32);
    v33 = (_QWORD *)((char *)v5 + v30 + 16);
    while ( 1 )
    {
      v35 = (_BYTE *)*((_QWORD *)v32 - 2);
      v36 = (__int64 *)*(v31 - 2);
      if ( v35 == v32 )
      {
        v37 = *((_QWORD *)v32 - 1);
        if ( v37 )
        {
          if ( v37 == 1 )
          {
            *(_BYTE *)v36 = *v32;
            v37 = *((_QWORD *)v32 - 1);
            v36 = (__int64 *)*(v31 - 2);
          }
          else
          {
            v38 = v33;
            v43 = v30;
            memcpy(v36, v32, v37);
            v37 = *((_QWORD *)v32 - 1);
            v36 = (__int64 *)*(v31 - 2);
            v33 = v38;
            v30 = v43;
          }
        }
        *(v31 - 1) = v37;
        *((_BYTE *)v36 + v37) = 0;
        v36 = (__int64 *)*((_QWORD *)v32 - 2);
        goto LABEL_60;
      }
      if ( v36 == v31 )
        break;
      *(v31 - 2) = (__int64)v35;
      v34 = *v31;
      *(v31 - 1) = *((_QWORD *)v32 - 1);
      *v31 = *(_QWORD *)v32;
      if ( !v36 )
        goto LABEL_67;
      *((_QWORD *)v32 - 2) = v36;
      *(_QWORD *)v32 = v34;
LABEL_60:
      v31 += 4;
      *((_QWORD *)v32 - 1) = 0;
      v32 += 32;
      *(_BYTE *)v36 = 0;
      if ( v33 == v31 )
      {
        v18 = *(_QWORD **)a1;
        v6 = *(unsigned int *)(a1 + 8);
        v7 = (_QWORD *)((char *)v5 + v30);
        goto LABEL_28;
      }
    }
    *(v31 - 2) = (__int64)v35;
    *(v31 - 1) = *((_QWORD *)v32 - 1);
    *v31 = *(_QWORD *)v32;
LABEL_67:
    *((_QWORD *)v32 - 2) = v32;
    v36 = (__int64 *)v32;
    goto LABEL_60;
  }
  if ( v8 > *(unsigned int *)(a1 + 12) )
  {
    v29 = &v5[4 * v6];
    while ( v29 != v5 )
    {
      while ( 1 )
      {
        v29 -= 4;
        if ( (_QWORD *)*v29 == v29 + 2 )
          break;
        v42 = v8;
        j_j___libc_free_0(*v29, v29[2] + 1LL);
        v8 = v42;
        if ( v29 == v5 )
          goto LABEL_54;
      }
    }
LABEL_54:
    *(_DWORD *)(a1 + 8) = 0;
    v6 = 0;
    sub_95D880(a1, v8);
    v3 = *(_QWORD *)a2;
    v8 = *(unsigned int *)(a2 + 8);
    v5 = *(_QWORD **)a1;
    v9 = *(const __m128i **)a2;
    goto LABEL_6;
  }
  v9 = (const __m128i *)(a2 + 16);
  if ( !*(_DWORD *)(a1 + 8) )
    goto LABEL_6;
  v6 *= 32LL;
  v22 = v5 + 2;
  v23 = (_BYTE *)(v4 + 32);
  v24 = (__int64 *)((char *)v22 + v6);
  do
  {
    v26 = (_BYTE *)*((_QWORD *)v23 - 2);
    v27 = (__int64 *)*(v22 - 2);
    if ( v26 == v23 )
    {
      v28 = *((_QWORD *)v23 - 1);
      if ( v28 )
      {
        if ( v28 == 1 )
        {
          *(_BYTE *)v27 = *v23;
          v28 = *((_QWORD *)v23 - 1);
          v27 = (__int64 *)*(v22 - 2);
        }
        else
        {
          v41 = v24;
          memcpy(v27, v23, v28);
          v28 = *((_QWORD *)v23 - 1);
          v27 = (__int64 *)*(v22 - 2);
          v24 = v41;
        }
      }
      *(v22 - 1) = v28;
      *((_BYTE *)v27 + v28) = 0;
      v27 = (__int64 *)*((_QWORD *)v23 - 2);
    }
    else
    {
      if ( v27 == v22 )
      {
        *(v22 - 2) = (__int64)v26;
        *(v22 - 1) = *((_QWORD *)v23 - 1);
        *v22 = *(_QWORD *)v23;
      }
      else
      {
        *(v22 - 2) = (__int64)v26;
        v25 = *v22;
        *(v22 - 1) = *((_QWORD *)v23 - 1);
        *v22 = *(_QWORD *)v23;
        if ( v27 )
        {
          *((_QWORD *)v23 - 2) = v27;
          *(_QWORD *)v23 = v25;
          goto LABEL_41;
        }
      }
      *((_QWORD *)v23 - 2) = v23;
      v27 = (__int64 *)v23;
    }
LABEL_41:
    v22 += 4;
    *((_QWORD *)v23 - 1) = 0;
    v23 += 32;
    *(_BYTE *)v27 = 0;
  }
  while ( v22 != v24 );
  v3 = *(_QWORD *)v4;
  v8 = *(unsigned int *)(v4 + 8);
  v5 = *(_QWORD **)a1;
  v9 = (const __m128i *)(*(_QWORD *)v4 + v6);
LABEL_6:
  v10 = (__m128i *)((char *)v5 + v6);
  v11 = v9 + 1;
  v12 = (const __m128i *)(32 * v8 + v3);
  v13 = (__m128i *)((char *)v5 + v6 + (char *)v12 - (char *)v9);
  if ( v12 != v9 )
  {
    do
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
        v14 = (const __m128i *)v11[-1].m128i_i64[0];
        if ( v14 == v11 )
        {
          v10[1] = _mm_loadu_si128(v11);
        }
        else
        {
          v10->m128i_i64[0] = (__int64)v14;
          v10[1].m128i_i64[0] = v11->m128i_i64[0];
        }
        v10->m128i_i64[1] = v11[-1].m128i_i64[1];
        v11[-1].m128i_i64[0] = (__int64)v11;
        v11[-1].m128i_i64[1] = 0;
        v11->m128i_i8[0] = 0;
      }
      v10 += 2;
      v11 += 2;
    }
    while ( v10 != v13 );
  }
  *(_DWORD *)(a1 + 8) = v39;
  v15 = *(_QWORD **)v4;
  v16 = (_QWORD *)(*(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8));
  if ( *(_QWORD **)v4 != v16 )
  {
    do
    {
      v16 -= 4;
      if ( (_QWORD *)*v16 != v16 + 2 )
        j_j___libc_free_0(*v16, v16[2] + 1LL);
    }
    while ( v15 != v16 );
  }
LABEL_18:
  *(_DWORD *)(v4 + 8) = 0;
}
