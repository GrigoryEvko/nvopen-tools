// Function: sub_1099B30
// Address: 0x1099b30
//
void __fastcall sub_1099B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  __m128i *v10; // r12
  unsigned __int64 v11; // rdx
  const __m128i *v12; // rsi
  const __m128i *v13; // rax
  const __m128i *v14; // r14
  __m128i *v15; // rcx
  const __m128i *v16; // rdx
  _QWORD *v17; // r12
  _QWORD *v18; // rbx
  __m128i *v19; // rbx
  __m128i *v20; // rax
  __m128i *v21; // rbx
  _QWORD *v22; // r12
  _QWORD *v23; // rbx
  __int64 v24; // rbx
  __m128i *v25; // r12
  __m128i *v26; // r14
  __m128i *v27; // rcx
  __int64 v28; // rdx
  __m128i *v29; // rax
  __m128i *v30; // rdi
  size_t v31; // rdx
  __m128i *v32; // rbx
  __m128i *v33; // rsi
  int v34; // r14d
  __m128i *v35; // r14
  __m128i *v36; // rbx
  __int64 v37; // r8
  __int64 m128i_i64; // rcx
  __int64 v39; // rdx
  __m128i *v40; // rax
  __m128i *v41; // rdi
  size_t v42; // rdx
  __int64 v43; // [rsp-60h] [rbp-60h]
  __m128i *v44; // [rsp-58h] [rbp-58h]
  unsigned __int64 v45; // [rsp-58h] [rbp-58h]
  __int64 v46; // [rsp-58h] [rbp-58h]
  int v47; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v48; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 == a2 )
    return;
  v7 = a2 + 16;
  v8 = a2;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(__m128i **)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v19 = &v10[2 * v9];
    if ( v19 != v10 )
    {
      do
      {
        v19 -= 2;
        if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
        {
          a2 = v19[1].m128i_i64[0] + 1;
          j_j___libc_free_0(v19->m128i_i64[0], a2);
        }
      }
      while ( v19 != v10 );
      v10 = *(__m128i **)a1;
    }
    if ( v10 != (__m128i *)(a1 + 16) )
      _libc_free(v10, a2);
    *(_QWORD *)a1 = *(_QWORD *)v8;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(v8 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(v8 + 12);
    *(_QWORD *)v8 = v7;
    *(_QWORD *)(v8 + 8) = 0;
    return;
  }
  v11 = *(unsigned int *)(a2 + 8);
  v47 = *(_DWORD *)(a2 + 8);
  if ( v11 > v9 )
  {
    if ( v11 > *(unsigned int *)(a1 + 12) )
    {
      v32 = &v10[2 * v9];
      while ( v32 != v10 )
      {
        while ( 1 )
        {
          v32 -= 2;
          if ( (__m128i *)v32->m128i_i64[0] == &v32[1] )
            break;
          v45 = v11;
          j_j___libc_free_0(v32->m128i_i64[0], v32[1].m128i_i64[0] + 1);
          v11 = v45;
          if ( v32 == v10 )
            goto LABEL_54;
        }
      }
LABEL_54:
      *(_DWORD *)(a1 + 8) = 0;
      v33 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v11, 0x20u, &v48, a6);
      v10 = v33;
      sub_1099A70(a1, v33);
      v34 = v48;
      if ( a1 + 16 != *(_QWORD *)a1 )
        _libc_free(*(_QWORD *)a1, v33);
      *(_QWORD *)a1 = v33;
      *(_DWORD *)(a1 + 12) = v34;
      v7 = *(_QWORD *)v8;
      v11 = *(unsigned int *)(v8 + 8);
      v12 = *(const __m128i **)v8;
LABEL_6:
      v13 = v12 + 1;
      v14 = (const __m128i *)(32 * v11 + v7);
      v15 = (__m128i *)((char *)v10 + (char *)v14 - (char *)v12);
      if ( v14 != v12 )
      {
        do
        {
          if ( v10 )
          {
            v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
            v16 = (const __m128i *)v13[-1].m128i_i64[0];
            if ( v16 == v13 )
            {
              v10[1] = _mm_loadu_si128(v13);
            }
            else
            {
              v10->m128i_i64[0] = (__int64)v16;
              v10[1].m128i_i64[0] = v13->m128i_i64[0];
            }
            v10->m128i_i64[1] = v13[-1].m128i_i64[1];
            v13[-1].m128i_i64[0] = (__int64)v13;
            v13[-1].m128i_i64[1] = 0;
            v13->m128i_i8[0] = 0;
          }
          v10 += 2;
          v13 += 2;
        }
        while ( v10 != v15 );
      }
      *(_DWORD *)(a1 + 8) = v47;
      v17 = *(_QWORD **)v8;
      v18 = (_QWORD *)(*(_QWORD *)v8 + 32LL * *(unsigned int *)(v8 + 8));
      if ( *(_QWORD **)v8 != v18 )
      {
        do
        {
          v18 -= 4;
          if ( (_QWORD *)*v18 != v18 + 2 )
            j_j___libc_free_0(*v18, v18[2] + 1LL);
        }
        while ( v17 != v18 );
      }
      goto LABEL_18;
    }
    v12 = (const __m128i *)(a2 + 16);
    if ( !*(_DWORD *)(a1 + 8) )
      goto LABEL_6;
    v24 = 32 * v9;
    v25 = v10 + 1;
    v26 = (__m128i *)(v8 + 32);
    v27 = &v25[(unsigned __int64)v24 / 0x10];
    while ( 1 )
    {
      v29 = (__m128i *)v26[-1].m128i_i64[0];
      v30 = (__m128i *)v25[-1].m128i_i64[0];
      if ( v29 == v26 )
      {
        v31 = v26[-1].m128i_u64[1];
        if ( v31 )
        {
          if ( v31 == 1 )
          {
            v30->m128i_i8[0] = v26->m128i_i8[0];
            v31 = v26[-1].m128i_u64[1];
            v30 = (__m128i *)v25[-1].m128i_i64[0];
          }
          else
          {
            v44 = v27;
            memcpy(v30, v26, v31);
            v31 = v26[-1].m128i_u64[1];
            v30 = (__m128i *)v25[-1].m128i_i64[0];
            v27 = v44;
          }
        }
        v25[-1].m128i_i64[1] = v31;
        v30->m128i_i8[v31] = 0;
        v30 = (__m128i *)v26[-1].m128i_i64[0];
        goto LABEL_41;
      }
      if ( v30 == v25 )
        break;
      v25[-1].m128i_i64[0] = (__int64)v29;
      v28 = v25->m128i_i64[0];
      v25[-1].m128i_i64[1] = v26[-1].m128i_i64[1];
      v25->m128i_i64[0] = v26->m128i_i64[0];
      if ( !v30 )
        goto LABEL_49;
      v26[-1].m128i_i64[0] = (__int64)v30;
      v26->m128i_i64[0] = v28;
LABEL_41:
      v25 += 2;
      v26[-1].m128i_i64[1] = 0;
      v26 += 2;
      v30->m128i_i8[0] = 0;
      if ( v25 == v27 )
      {
        v7 = *(_QWORD *)v8;
        v11 = *(unsigned int *)(v8 + 8);
        v10 = (__m128i *)(v24 + *(_QWORD *)a1);
        v12 = (const __m128i *)(*(_QWORD *)v8 + v24);
        goto LABEL_6;
      }
    }
    v25[-1].m128i_i64[0] = (__int64)v29;
    v25[-1].m128i_i64[1] = v26[-1].m128i_i64[1];
    v25->m128i_i64[0] = v26->m128i_i64[0];
LABEL_49:
    v26[-1].m128i_i64[0] = (__int64)v26;
    v30 = v26;
    goto LABEL_41;
  }
  v20 = *(__m128i **)a1;
  if ( !*(_DWORD *)(a2 + 8) )
    goto LABEL_28;
  v35 = v10 + 1;
  v36 = (__m128i *)(a2 + 32);
  v37 = 32 * v11;
  m128i_i64 = (__int64)v10[2 * v11 + 1].m128i_i64;
  do
  {
    v40 = (__m128i *)v36[-1].m128i_i64[0];
    v41 = (__m128i *)v35[-1].m128i_i64[0];
    if ( v40 == v36 )
    {
      v42 = v36[-1].m128i_u64[1];
      if ( v42 )
      {
        if ( v42 == 1 )
        {
          v41->m128i_i8[0] = v36->m128i_i8[0];
          v42 = v36[-1].m128i_u64[1];
          v41 = (__m128i *)v35[-1].m128i_i64[0];
        }
        else
        {
          v43 = m128i_i64;
          v46 = v37;
          memcpy(v41, v36, v42);
          v42 = v36[-1].m128i_u64[1];
          v41 = (__m128i *)v35[-1].m128i_i64[0];
          m128i_i64 = v43;
          v37 = v46;
        }
      }
      v35[-1].m128i_i64[1] = v42;
      v41->m128i_i8[v42] = 0;
      v41 = (__m128i *)v36[-1].m128i_i64[0];
    }
    else
    {
      if ( v41 == v35 )
      {
        v35[-1].m128i_i64[0] = (__int64)v40;
        v35[-1].m128i_i64[1] = v36[-1].m128i_i64[1];
        v35->m128i_i64[0] = v36->m128i_i64[0];
      }
      else
      {
        v35[-1].m128i_i64[0] = (__int64)v40;
        v39 = v35->m128i_i64[0];
        v35[-1].m128i_i64[1] = v36[-1].m128i_i64[1];
        v35->m128i_i64[0] = v36->m128i_i64[0];
        if ( v41 )
        {
          v36[-1].m128i_i64[0] = (__int64)v41;
          v36->m128i_i64[0] = v39;
          goto LABEL_62;
        }
      }
      v36[-1].m128i_i64[0] = (__int64)v36;
      v41 = v36;
    }
LABEL_62:
    v35 += 2;
    v36[-1].m128i_i64[1] = 0;
    v36 += 2;
    v41->m128i_i8[0] = 0;
  }
  while ( (__m128i *)m128i_i64 != v35 );
  v20 = *(__m128i **)a1;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = (__m128i *)((char *)v10 + v37);
LABEL_28:
  v21 = &v20[2 * v9];
  while ( v10 != v21 )
  {
    v21 -= 2;
    if ( (__m128i *)v21->m128i_i64[0] != &v21[1] )
      j_j___libc_free_0(v21->m128i_i64[0], v21[1].m128i_i64[0] + 1);
  }
  *(_DWORD *)(a1 + 8) = v47;
  v22 = *(_QWORD **)a2;
  v23 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
  if ( *(_QWORD **)a2 != v23 )
  {
    do
    {
      v23 -= 4;
      if ( (_QWORD *)*v23 != v23 + 2 )
        j_j___libc_free_0(*v23, v23[2] + 1LL);
    }
    while ( v22 != v23 );
  }
LABEL_18:
  *(_DWORD *)(v8 + 8) = 0;
}
