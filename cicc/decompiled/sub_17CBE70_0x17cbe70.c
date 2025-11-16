// Function: sub_17CBE70
// Address: 0x17cbe70
//
unsigned __int64 __fastcall sub_17CBE70(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 result; // rax
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // r11d
  unsigned int v12; // r13d
  unsigned int v13; // r9d
  __int64 *v14; // rdx
  __int64 v15; // r10
  _QWORD *v16; // rdi
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  __int64 v23; // r15
  int v24; // edi
  int v25; // ecx
  __int64 *v26; // rax
  int v27; // eax
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdi
  __int64 *v31; // r8
  unsigned int v32; // r13d
  int v33; // r9d
  __int64 v34; // rsi
  int v35; // r15d
  __int64 v36; // rdi
  int v37; // r10d
  __int64 *v38; // r9
  int v39; // [rsp+8h] [rbp-58h]
  unsigned int v40; // [rsp+Ch] [rbp-54h]
  __m128i v41; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v42; // [rsp+20h] [rbp-40h]

  v3 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v5 = *(_QWORD *)(a2 + 24 * (3 - v4));
  if ( *(_DWORD *)(v5 + 32) <= 0x40u )
    v6 = *(_QWORD *)(v5 + 24);
  else
    v6 = **(_QWORD **)(v5 + 24);
  v7 = *(_QWORD *)(a2 + 24 * (4 - v4));
  result = *(_QWORD *)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    result = *(_QWORD *)result;
  v9 = *(_DWORD *)(a1 + 136);
  v10 = *(_QWORD *)(a1 + 120);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 112);
    v41 = 0u;
    v42 = 0;
    v41.m128i_i32[v6] = result + 1;
    goto LABEL_12;
  }
  v11 = v9 - 1;
  v12 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v13 = (v9 - 1) & v12;
  v14 = (__int64 *)(v10 + 32LL * v13);
  v15 = *v14;
  v16 = v14;
  if ( v3 != *v14 )
  {
    v40 = (v9 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v23 = *v14;
    v24 = 1;
    while ( v23 != -8 )
    {
      v35 = v24 + 1;
      v36 = v11 & (v40 + v24);
      v39 = v35;
      v40 = v36;
      v16 = (_QWORD *)(v10 + 32 * v36);
      v23 = *v16;
      if ( v3 == *v16 )
        goto LABEL_7;
      v24 = v39;
    }
    v41.m128i_i64[0] = 0;
    v41.m128i_i32[v6] = result + 1;
    v41.m128i_i64[1] = 0;
    v12 = ((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9);
    v42 = 0;
    v13 = v12 & v11;
    v14 = (__int64 *)(v10 + 32LL * (v12 & v11));
    v15 = *v14;
LABEL_19:
    if ( v3 == v15 )
    {
LABEL_20:
      result = v42;
      *(__m128i *)(v14 + 1) = _mm_loadu_si128(&v41);
      v14[3] = result;
      return result;
    }
    v25 = 1;
    v26 = 0;
    while ( v15 != -8 )
    {
      if ( !v26 && v15 == -16 )
        v26 = v14;
      v13 = v11 & (v25 + v13);
      v14 = (__int64 *)(v10 + 32LL * v13);
      v15 = *v14;
      if ( v3 == *v14 )
        goto LABEL_20;
      ++v25;
    }
    if ( v26 )
      v14 = v26;
    v27 = *(_DWORD *)(a1 + 128);
    ++*(_QWORD *)(a1 + 112);
    v21 = v27 + 1;
    if ( 4 * (v27 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 132) - v21 > v9 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 128) = v21;
        if ( *v14 != -8 )
          --*(_DWORD *)(a1 + 132);
        *v14 = v3;
        v14[2] = 0;
        v14[3] = 0;
        v14[1] = 0;
        goto LABEL_20;
      }
      sub_17CA490(a1 + 112, v9);
      v28 = *(_DWORD *)(a1 + 136);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 120);
        v31 = 0;
        v32 = v29 & v12;
        v33 = 1;
        v21 = *(_DWORD *)(a1 + 128) + 1;
        v14 = (__int64 *)(v30 + 32LL * v32);
        v34 = *v14;
        if ( v3 != *v14 )
        {
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v31 )
              v31 = v14;
            v32 = v29 & (v33 + v32);
            v14 = (__int64 *)(v30 + 32LL * v32);
            v34 = *v14;
            if ( v3 == *v14 )
              goto LABEL_14;
            ++v33;
          }
          if ( v31 )
            v14 = v31;
        }
        goto LABEL_14;
      }
LABEL_58:
      ++*(_DWORD *)(a1 + 128);
      BUG();
    }
LABEL_12:
    sub_17CA490(a1 + 112, 2 * v9);
    v17 = *(_DWORD *)(a1 + 136);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 120);
      v20 = (v17 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v21 = *(_DWORD *)(a1 + 128) + 1;
      v14 = (__int64 *)(v19 + 32LL * v20);
      v22 = *v14;
      if ( v3 != *v14 )
      {
        v37 = 1;
        v38 = 0;
        while ( v22 != -8 )
        {
          if ( v22 == -16 && !v38 )
            v38 = v14;
          v20 = v18 & (v37 + v20);
          v14 = (__int64 *)(v19 + 32LL * v20);
          v22 = *v14;
          if ( v3 == *v14 )
            goto LABEL_14;
          ++v37;
        }
        if ( v38 )
          v14 = v38;
      }
      goto LABEL_14;
    }
    goto LABEL_58;
  }
LABEL_7:
  if ( v16 == (_QWORD *)(v10 + 32LL * v9) )
  {
    v41 = 0u;
    v42 = 0;
    v41.m128i_i32[v6] = result + 1;
    goto LABEL_19;
  }
  if ( *((unsigned int *)v16 + v6 + 2) <= result )
  {
    result = (unsigned int)(result + 1);
    *((_DWORD *)v16 + v6 + 2) = result;
  }
  return result;
}
