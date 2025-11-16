// Function: sub_2FD7D90
// Address: 0x2fd7d90
//
void __fastcall sub_2FD7D90(__int64 a1, int a2, __int32 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  unsigned int v10; // edx
  int *v11; // rdi
  int v12; // r9d
  __m128i *v13; // rsi
  int v14; // edi
  __int64 v15; // r13
  const __m128i *v16; // rdx
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __m128i *v20; // rax
  __m128i *v21; // r13
  unsigned int v22; // esi
  __int64 v23; // r9
  __int64 v24; // r8
  unsigned int v25; // edi
  _DWORD *v26; // rax
  int v27; // ecx
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  int v30; // r11d
  _DWORD *v31; // rdx
  int v32; // eax
  int v33; // ecx
  int v34; // eax
  int v35; // esi
  unsigned int v36; // edi
  int v37; // eax
  int v38; // r10d
  int v39; // eax
  int v40; // esi
  __int64 v41; // rdi
  int v42; // r10d
  int v43; // eax
  int v44; // [rsp+Ch] [rbp-64h]
  __m128i v45; // [rsp+10h] [rbp-60h] BYREF
  __m128i v46; // [rsp+20h] [rbp-50h] BYREF
  __int64 v47; // [rsp+30h] [rbp-40h]

  v8 = *(unsigned int *)(a1 + 168);
  v9 = *(_QWORD *)(a1 + 152);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (37 * a2);
    v11 = (int *)(v9 + 32LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
    {
LABEL_3:
      if ( v11 != (int *)(v9 + 32 * v8) )
      {
        v46.m128i_i64[0] = a4;
        v46.m128i_i32[2] = a3;
        v13 = (__m128i *)*((_QWORD *)v11 + 2);
        if ( v13 == *((__m128i **)v11 + 3) )
        {
          sub_2FD79F0((unsigned __int64 *)v11 + 1, v13, &v46);
        }
        else
        {
          if ( v13 )
          {
            *v13 = _mm_loadu_si128(&v46);
            v13 = (__m128i *)*((_QWORD *)v11 + 2);
          }
          *((_QWORD *)v11 + 2) = v13 + 1;
        }
        return;
      }
    }
    else
    {
      v14 = 1;
      while ( v12 != -1 )
      {
        v30 = v14 + 1;
        v10 = (v8 - 1) & (v14 + v10);
        v11 = (int *)(v9 + 32LL * v10);
        v12 = *v11;
        if ( *v11 == a2 )
          goto LABEL_3;
        v14 = v30;
      }
    }
  }
  v45.m128i_i32[2] = a3;
  v46 = 0u;
  v47 = 0;
  v45.m128i_i64[0] = a4;
  sub_2FD79F0((unsigned __int64 *)&v46, 0, &v45);
  v15 = v46.m128i_i64[1];
  v16 = (const __m128i *)v46.m128i_i64[0];
  v17 = v46.m128i_i64[1] - v46.m128i_i64[0];
  if ( v46.m128i_i64[1] == v46.m128i_i64[0] )
  {
    v19 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(&v46, 0, v46.m128i_i64[0]);
    v18 = sub_22077B0(v46.m128i_i64[1] - v46.m128i_i64[0]);
    v15 = v46.m128i_i64[1];
    v16 = (const __m128i *)v46.m128i_i64[0];
    v19 = v18;
  }
  if ( (const __m128i *)v15 == v16 )
  {
    v21 = (__m128i *)v19;
  }
  else
  {
    v20 = (__m128i *)v19;
    v21 = (__m128i *)(v19 + v15 - (_QWORD)v16);
    do
    {
      if ( v20 )
        *v20 = _mm_loadu_si128(v16);
      ++v20;
      ++v16;
    }
    while ( v20 != v21 );
  }
  v22 = *(_DWORD *)(a1 + 168);
  if ( !v22 )
  {
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_42;
  }
  v23 = *(_QWORD *)(a1 + 152);
  v24 = (unsigned int)(37 * a2);
  v25 = (v22 - 1) & (37 * a2);
  v26 = (_DWORD *)(v23 + 32LL * v25);
  v27 = *v26;
  if ( *v26 != a2 )
  {
    v44 = 1;
    v31 = 0;
    while ( v27 != -1 )
    {
      if ( v27 != -2 || v31 )
        v26 = v31;
      v25 = (v22 - 1) & (v44 + v25);
      v27 = *(_DWORD *)(v23 + 32LL * v25);
      if ( v27 == a2 )
        goto LABEL_21;
      ++v44;
      v31 = v26;
      v26 = (_DWORD *)(v23 + 32LL * v25);
    }
    if ( !v31 )
      v31 = v26;
    v32 = *(_DWORD *)(a1 + 160);
    ++*(_QWORD *)(a1 + 144);
    v33 = v32 + 1;
    if ( 4 * (v32 + 1) < 3 * v22 )
    {
      if ( v22 - *(_DWORD *)(a1 + 164) - v33 > v22 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(a1 + 160) = v33;
        if ( *v31 != -1 )
          --*(_DWORD *)(a1 + 164);
        *((_QWORD *)v31 + 1) = v19;
        *v31 = a2;
        *((_QWORD *)v31 + 2) = v21;
        *((_QWORD *)v31 + 3) = v17 + v19;
        goto LABEL_23;
      }
      sub_2FD7B70(a1 + 144, v22);
      v39 = *(_DWORD *)(a1 + 168);
      if ( v39 )
      {
        v40 = v39 - 1;
        v23 = 0;
        v41 = *(_QWORD *)(a1 + 152);
        v42 = 1;
        v24 = (v39 - 1) & (unsigned int)(37 * a2);
        v33 = *(_DWORD *)(a1 + 160) + 1;
        v31 = (_DWORD *)(v41 + 32 * v24);
        v43 = *v31;
        if ( *v31 == a2 )
          goto LABEL_38;
        while ( v43 != -1 )
        {
          if ( v43 == -2 && !v23 )
            v23 = (__int64)v31;
          v24 = v40 & (unsigned int)(v42 + v24);
          v31 = (_DWORD *)(v41 + 32LL * (unsigned int)v24);
          v43 = *v31;
          if ( *v31 == a2 )
            goto LABEL_38;
          ++v42;
        }
        goto LABEL_46;
      }
      goto LABEL_68;
    }
LABEL_42:
    sub_2FD7B70(a1 + 144, 2 * v22);
    v34 = *(_DWORD *)(a1 + 168);
    if ( v34 )
    {
      v35 = v34 - 1;
      v24 = *(_QWORD *)(a1 + 152);
      v36 = (v34 - 1) & (37 * a2);
      v33 = *(_DWORD *)(a1 + 160) + 1;
      v31 = (_DWORD *)(v24 + 32LL * v36);
      v37 = *v31;
      if ( *v31 == a2 )
        goto LABEL_38;
      v38 = 1;
      v23 = 0;
      while ( v37 != -1 )
      {
        if ( !v23 && v37 == -2 )
          v23 = (__int64)v31;
        v36 = v35 & (v38 + v36);
        v31 = (_DWORD *)(v24 + 32LL * v36);
        v37 = *v31;
        if ( *v31 == a2 )
          goto LABEL_38;
        ++v38;
      }
LABEL_46:
      if ( v23 )
        v31 = (_DWORD *)v23;
      goto LABEL_38;
    }
LABEL_68:
    ++*(_DWORD *)(a1 + 160);
    BUG();
  }
LABEL_21:
  if ( v19 )
    j_j___libc_free_0(v19);
LABEL_23:
  v28 = *(unsigned int *)(a1 + 72);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
  {
    sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v28 + 1, 4u, v24, v23);
    v28 = *(unsigned int *)(a1 + 72);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 64) + 4 * v28) = a2;
  v29 = v46.m128i_i64[0];
  ++*(_DWORD *)(a1 + 72);
  if ( v29 )
    j_j___libc_free_0(v29);
}
