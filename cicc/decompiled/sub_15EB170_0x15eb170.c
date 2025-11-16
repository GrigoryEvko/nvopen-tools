// Function: sub_15EB170
// Address: 0x15eb170
//
void __fastcall sub_15EB170(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const __m128i *v4; // r14
  _QWORD *v6; // r15
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r8
  _QWORD *v9; // rbx
  unsigned __int64 v10; // r9
  const __m128i *v11; // rdx
  __m128i *v12; // rbx
  const __m128i *v13; // rax
  const __m128i *v14; // r14
  __m128i *v15; // rcx
  const __m128i *v16; // rdx
  _QWORD *v17; // rbx
  const __m128i *v18; // r13
  const __m128i *v19; // rbx
  unsigned __int64 v20; // rax
  _QWORD *v21; // rbx
  const __m128i *v22; // r13
  const __m128i *v23; // rbx
  __int64 v24; // r9
  __int64 *v25; // r14
  _BYTE *v26; // rbx
  __int64 *v27; // r8
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 *v30; // rdi
  size_t v31; // rdx
  _QWORD *v32; // rcx
  _BYTE *v33; // r14
  _QWORD *v34; // r15
  __int64 v35; // rdx
  _BYTE *v36; // rax
  _BYTE *v37; // rdi
  size_t v38; // rdx
  __int64 *v39; // [rsp-50h] [rbp-50h]
  int v40; // [rsp-44h] [rbp-44h]
  unsigned __int64 v41; // [rsp-40h] [rbp-40h]
  unsigned __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  _QWORD *v44; // [rsp-40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v4 = (const __m128i *)(a2 + 16);
  v6 = *(_QWORD **)a1;
  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v9 = &v6[4 * v7];
    if ( v9 != v6 )
    {
      do
      {
        v9 -= 4;
        if ( (_QWORD *)*v9 != v9 + 2 )
          j_j___libc_free_0(*v9, v9[2] + 1LL);
      }
      while ( v9 != v6 );
      v8 = *(_QWORD *)a1;
    }
    if ( v8 != a1 + 16 )
      _libc_free(v8);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v4;
    *(_QWORD *)(a2 + 8) = 0;
    return;
  }
  v10 = *(unsigned int *)(a2 + 8);
  v40 = *(_DWORD *)(a2 + 8);
  if ( v10 > v7 )
  {
    if ( v10 > *(unsigned int *)(a1 + 12) )
    {
      v17 = &v6[4 * v7];
      while ( v17 != v6 )
      {
        while ( 1 )
        {
          v17 -= 4;
          if ( (_QWORD *)*v17 == v17 + 2 )
            break;
          v41 = v10;
          j_j___libc_free_0(*v17, v17[2] + 1LL);
          v10 = v41;
          if ( v17 == v6 )
            goto LABEL_26;
        }
      }
LABEL_26:
      *(_DWORD *)(a1 + 8) = 0;
      v7 = 0;
      sub_12BE710(a1, v10, a3, a4, v8, v10);
      v4 = *(const __m128i **)a2;
      v10 = *(unsigned int *)(a2 + 8);
      v6 = *(_QWORD **)a1;
      v11 = *(const __m128i **)a2;
      goto LABEL_14;
    }
    v11 = (const __m128i *)(a2 + 16);
    if ( !*(_DWORD *)(a1 + 8) )
      goto LABEL_14;
    v7 *= 32LL;
    v32 = v6 + 2;
    v33 = (_BYTE *)(a2 + 32);
    v34 = (_QWORD *)((char *)v6 + v7 + 16);
    while ( 1 )
    {
      v36 = (_BYTE *)*((_QWORD *)v33 - 2);
      v37 = (_BYTE *)*(v32 - 2);
      if ( v36 == v33 )
      {
        v38 = *((_QWORD *)v33 - 1);
        if ( v38 )
        {
          if ( v38 == 1 )
          {
            *v37 = *v33;
            v38 = *((_QWORD *)v33 - 1);
            v37 = (_BYTE *)*(v32 - 2);
          }
          else
          {
            v44 = v32;
            memcpy(v37, v33, v38);
            v32 = v44;
            v38 = *((_QWORD *)v33 - 1);
            v37 = (_BYTE *)*(v44 - 2);
          }
        }
        *(v32 - 1) = v38;
        v37[v38] = 0;
        v37 = (_BYTE *)*((_QWORD *)v33 - 2);
        goto LABEL_61;
      }
      if ( v37 == (_BYTE *)v32 )
        break;
      *(v32 - 2) = v36;
      v35 = *v32;
      *(v32 - 1) = *((_QWORD *)v33 - 1);
      *v32 = *(_QWORD *)v33;
      if ( !v37 )
        goto LABEL_68;
      *((_QWORD *)v33 - 2) = v37;
      *(_QWORD *)v33 = v35;
LABEL_61:
      v32 += 4;
      *((_QWORD *)v33 - 1) = 0;
      v33 += 32;
      *v37 = 0;
      if ( v32 == v34 )
      {
        v4 = *(const __m128i **)a2;
        v10 = *(unsigned int *)(a2 + 8);
        v6 = *(_QWORD **)a1;
        v11 = (const __m128i *)(*(_QWORD *)a2 + v7);
LABEL_14:
        v12 = (__m128i *)((char *)v6 + v7);
        v13 = v11 + 1;
        v14 = &v4[2 * v10];
        v15 = (__m128i *)((char *)v12 + (char *)v14 - (char *)v11);
        if ( v14 != v11 )
        {
          do
          {
            if ( v12 )
            {
              v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
              v16 = (const __m128i *)v13[-1].m128i_i64[0];
              if ( v16 == v13 )
              {
                v12[1] = _mm_loadu_si128(v13);
              }
              else
              {
                v12->m128i_i64[0] = (__int64)v16;
                v12[1].m128i_i64[0] = v13->m128i_i64[0];
              }
              v12->m128i_i64[1] = v13[-1].m128i_i64[1];
              v13[-1].m128i_i64[0] = (__int64)v13;
              v13[-1].m128i_i64[1] = 0;
              v13->m128i_i8[0] = 0;
            }
            v12 += 2;
            v13 += 2;
          }
          while ( v12 != v15 );
        }
        *(_DWORD *)(a1 + 8) = v40;
        v18 = *(const __m128i **)a2;
        v19 = (const __m128i *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(const __m128i **)a2 != v19 )
        {
          do
          {
            v19 -= 2;
            if ( (const __m128i *)v19->m128i_i64[0] != &v19[1] )
              j_j___libc_free_0(v19->m128i_i64[0], v19[1].m128i_i64[0] + 1);
          }
          while ( v18 != v19 );
        }
LABEL_31:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
    }
    *(v32 - 2) = v36;
    *(v32 - 1) = *((_QWORD *)v33 - 1);
    *v32 = *(_QWORD *)v33;
LABEL_68:
    *((_QWORD *)v33 - 2) = v33;
    v37 = v33;
    goto LABEL_61;
  }
  v20 = *(_QWORD *)a1;
  if ( *(_DWORD *)(a2 + 8) )
  {
    v24 = 32 * v10;
    v25 = v6 + 2;
    v26 = (_BYTE *)(a2 + 32);
    v27 = (_QWORD *)((char *)v6 + v24 + 16);
    while ( 1 )
    {
      v29 = (_BYTE *)*((_QWORD *)v26 - 2);
      v30 = (__int64 *)*(v25 - 2);
      if ( v29 == v26 )
      {
        v31 = *((_QWORD *)v26 - 1);
        if ( v31 )
        {
          if ( v31 == 1 )
          {
            *(_BYTE *)v30 = *v26;
            v31 = *((_QWORD *)v26 - 1);
            v30 = (__int64 *)*(v25 - 2);
          }
          else
          {
            v39 = v27;
            v43 = v24;
            memcpy(v30, v26, v31);
            v31 = *((_QWORD *)v26 - 1);
            v30 = (__int64 *)*(v25 - 2);
            v27 = v39;
            v24 = v43;
          }
        }
        *(v25 - 1) = v31;
        *((_BYTE *)v30 + v31) = 0;
        v30 = (__int64 *)*((_QWORD *)v26 - 2);
        goto LABEL_48;
      }
      if ( v25 == v30 )
        break;
      *(v25 - 2) = (__int64)v29;
      v28 = *v25;
      *(v25 - 1) = *((_QWORD *)v26 - 1);
      *v25 = *(_QWORD *)v26;
      if ( !v30 )
        goto LABEL_55;
      *((_QWORD *)v26 - 2) = v30;
      *(_QWORD *)v26 = v28;
LABEL_48:
      v25 += 4;
      *((_QWORD *)v26 - 1) = 0;
      v26 += 32;
      *(_BYTE *)v30 = 0;
      if ( v27 == v25 )
      {
        v20 = *(_QWORD *)a1;
        v7 = *(unsigned int *)(a1 + 8);
        v8 = (unsigned __int64)v6 + v24;
        goto LABEL_35;
      }
    }
    *(v25 - 2) = (__int64)v29;
    *(v25 - 1) = *((_QWORD *)v26 - 1);
    *v25 = *(_QWORD *)v26;
LABEL_55:
    *((_QWORD *)v26 - 2) = v26;
    v30 = (__int64 *)v26;
    goto LABEL_48;
  }
LABEL_35:
  v21 = (_QWORD *)(v20 + 32 * v7);
  while ( (_QWORD *)v8 != v21 )
  {
    v21 -= 4;
    if ( (_QWORD *)*v21 != v21 + 2 )
    {
      v42 = v8;
      j_j___libc_free_0(*v21, v21[2] + 1LL);
      v8 = v42;
    }
  }
  *(_DWORD *)(a1 + 8) = v40;
  v22 = *(const __m128i **)a2;
  v23 = (const __m128i *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
  if ( *(const __m128i **)a2 == v23 )
    goto LABEL_31;
  do
  {
    v23 -= 2;
    if ( (const __m128i *)v23->m128i_i64[0] != &v23[1] )
      j_j___libc_free_0(v23->m128i_i64[0], v23[1].m128i_i64[0] + 1);
  }
  while ( v22 != v23 );
  *(_DWORD *)(a2 + 8) = 0;
}
