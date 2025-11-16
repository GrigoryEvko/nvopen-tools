// Function: sub_1F357A0
// Address: 0x1f357a0
//
__int64 __fastcall sub_1F357A0(__int64 a1, int a2, __int32 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned int v9; // edx
  int *v10; // rdi
  int v11; // r9d
  __int64 result; // rax
  __m128i *v13; // rsi
  int v14; // edi
  __int64 v15; // r13
  const __m128i *v16; // rdx
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  __m128i *v19; // r14
  __m128i *v20; // rax
  __m128i *v21; // r13
  unsigned int v22; // esi
  _DWORD *v23; // r9
  __int64 v24; // r8
  unsigned int v25; // edi
  _DWORD *v26; // rax
  int v27; // ecx
  __int64 v28; // rdi
  int v29; // r11d
  _DWORD *v30; // rdx
  int v31; // eax
  int v32; // ecx
  int v33; // eax
  int v34; // edi
  unsigned int v35; // eax
  int v36; // esi
  int v37; // r10d
  int v38; // eax
  int v39; // esi
  __int64 v40; // rdi
  int v41; // r10d
  int v42; // eax
  int v43; // [rsp+Ch] [rbp-64h]
  __m128i v44; // [rsp+10h] [rbp-60h] BYREF
  __m128i v45; // [rsp+20h] [rbp-50h] BYREF
  __int64 v46; // [rsp+30h] [rbp-40h]

  v7 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 144);
    v9 = (v7 - 1) & (37 * a2);
    v10 = (int *)(v8 + 32LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
    {
LABEL_3:
      result = v8 + 32 * v7;
      if ( v10 != (int *)result )
      {
        v45.m128i_i64[0] = a4;
        v45.m128i_i32[2] = a3;
        v13 = (__m128i *)*((_QWORD *)v10 + 2);
        if ( v13 == *((__m128i **)v10 + 3) )
          return sub_1F35400((const __m128i **)v10 + 1, v13, &v45);
        if ( v13 )
        {
          *v13 = _mm_loadu_si128(&v45);
          v13 = (__m128i *)*((_QWORD *)v10 + 2);
        }
        *((_QWORD *)v10 + 2) = v13 + 1;
        return result;
      }
    }
    else
    {
      v14 = 1;
      while ( v11 != -1 )
      {
        v29 = v14 + 1;
        v9 = (v7 - 1) & (v14 + v9);
        v10 = (int *)(v8 + 32LL * v9);
        v11 = *v10;
        if ( *v10 == a2 )
          goto LABEL_3;
        v14 = v29;
      }
    }
  }
  v44.m128i_i32[2] = a3;
  v45 = 0u;
  v46 = 0;
  v44.m128i_i64[0] = a4;
  sub_1F35400((const __m128i **)&v45, 0, &v44);
  v15 = v45.m128i_i64[1];
  v16 = (const __m128i *)v45.m128i_i64[0];
  v17 = v45.m128i_i64[1] - v45.m128i_i64[0];
  if ( v45.m128i_i64[1] == v45.m128i_i64[0] )
  {
    v19 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(&v45, 0, v45.m128i_i64[0]);
    v18 = sub_22077B0(v45.m128i_i64[1] - v45.m128i_i64[0]);
    v15 = v45.m128i_i64[1];
    v16 = (const __m128i *)v45.m128i_i64[0];
    v19 = (__m128i *)v18;
  }
  if ( (const __m128i *)v15 == v16 )
  {
    v21 = v19;
  }
  else
  {
    v20 = v19;
    v21 = (__m128i *)((char *)v19 + v15 - (_QWORD)v16);
    do
    {
      if ( v20 )
        *v20 = _mm_loadu_si128(v16);
      ++v20;
      ++v16;
    }
    while ( v21 != v20 );
  }
  v22 = *(_DWORD *)(a1 + 160);
  if ( !v22 )
  {
    ++*(_QWORD *)(a1 + 136);
    goto LABEL_42;
  }
  v23 = *(_DWORD **)(a1 + 144);
  LODWORD(v24) = 37 * a2;
  v25 = (v22 - 1) & (37 * a2);
  v26 = &v23[8 * v25];
  v27 = *v26;
  if ( *v26 != a2 )
  {
    v43 = 1;
    v30 = 0;
    while ( v27 != -1 )
    {
      if ( v30 || v27 != -2 )
        v26 = v30;
      v25 = (v22 - 1) & (v43 + v25);
      v27 = v23[8 * v25];
      if ( v27 == a2 )
        goto LABEL_21;
      ++v43;
      v30 = v26;
      v26 = &v23[8 * v25];
    }
    if ( !v30 )
      v30 = v26;
    v31 = *(_DWORD *)(a1 + 152);
    ++*(_QWORD *)(a1 + 136);
    v32 = v31 + 1;
    if ( 4 * (v31 + 1) < 3 * v22 )
    {
      if ( v22 - *(_DWORD *)(a1 + 156) - v32 > v22 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(a1 + 152) = v32;
        if ( *v30 != -1 )
          --*(_DWORD *)(a1 + 156);
        *((_QWORD *)v30 + 1) = v19;
        *v30 = a2;
        *((_QWORD *)v30 + 2) = v21;
        *((_QWORD *)v30 + 3) = (char *)v19 + v17;
        goto LABEL_23;
      }
      sub_1F35580(a1 + 136, v22);
      v38 = *(_DWORD *)(a1 + 160);
      if ( v38 )
      {
        v39 = v38 - 1;
        v23 = 0;
        v40 = *(_QWORD *)(a1 + 144);
        v41 = 1;
        LODWORD(v24) = (v38 - 1) & (37 * a2);
        v32 = *(_DWORD *)(a1 + 152) + 1;
        v30 = (_DWORD *)(v40 + 32LL * (unsigned int)v24);
        v42 = *v30;
        if ( *v30 == a2 )
          goto LABEL_38;
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v23 )
            v23 = v30;
          LODWORD(v24) = v39 & (v41 + v24);
          v30 = (_DWORD *)(v40 + 32LL * (unsigned int)v24);
          v42 = *v30;
          if ( *v30 == a2 )
            goto LABEL_38;
          ++v41;
        }
        goto LABEL_46;
      }
      goto LABEL_68;
    }
LABEL_42:
    sub_1F35580(a1 + 136, 2 * v22);
    v33 = *(_DWORD *)(a1 + 160);
    if ( v33 )
    {
      v34 = v33 - 1;
      v24 = *(_QWORD *)(a1 + 144);
      v35 = (v33 - 1) & (37 * a2);
      v32 = *(_DWORD *)(a1 + 152) + 1;
      v30 = (_DWORD *)(v24 + 32LL * v35);
      v36 = *v30;
      if ( *v30 == a2 )
        goto LABEL_38;
      v37 = 1;
      v23 = 0;
      while ( v36 != -1 )
      {
        if ( v36 == -2 && !v23 )
          v23 = v30;
        v35 = v34 & (v37 + v35);
        v30 = (_DWORD *)(v24 + 32LL * v35);
        v36 = *v30;
        if ( *v30 == a2 )
          goto LABEL_38;
        ++v37;
      }
LABEL_46:
      if ( v23 )
        v30 = v23;
      goto LABEL_38;
    }
LABEL_68:
    ++*(_DWORD *)(a1 + 152);
    BUG();
  }
LABEL_21:
  if ( v19 )
    j_j___libc_free_0(v19, v17);
LABEL_23:
  result = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 68) )
  {
    sub_16CD150(a1 + 56, (const void *)(a1 + 72), 0, 4, v24, (int)v23);
    result = *(unsigned int *)(a1 + 64);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 56) + 4 * result) = a2;
  v28 = v45.m128i_i64[0];
  ++*(_DWORD *)(a1 + 64);
  if ( v28 )
    return j_j___libc_free_0(v28, v46 - v28);
  return result;
}
