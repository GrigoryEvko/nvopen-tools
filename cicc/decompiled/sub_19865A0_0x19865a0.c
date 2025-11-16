// Function: sub_19865A0
// Address: 0x19865a0
//
__int64 __fastcall sub_19865A0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rsi
  int v24; // r9d
  __int64 *v25; // r8
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  unsigned int v30; // r15d
  __int64 *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // [rsp+0h] [rbp-50h] BYREF
  __m128i v34; // [rsp+8h] [rbp-48h] BYREF
  int v35; // [rsp+18h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
  }
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_24:
    sub_14672C0(a1, 2 * v5);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v8;
      if ( v4 != *v8 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v8;
          v22 = v20 & (v24 + v22);
          v8 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v8 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_14672C0(a1, v5);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = 1;
      v30 = v27 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v31 = 0;
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v28 + 16LL * v30);
      v32 = *v8;
      if ( v4 != *v8 )
      {
        while ( v32 != -8 )
        {
          if ( !v31 && v32 == -16 )
            v31 = v8;
          v30 = v27 & (v29 + v30);
          v8 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v29;
        }
        if ( v31 )
          v8 = v31;
      }
      goto LABEL_15;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  v16 = *a2;
  v34 = 0u;
  v17 = *(_QWORD *)(a1 + 40);
  v33 = v16;
  v35 = 0;
  if ( v17 == *(_QWORD *)(a1 + 48) )
  {
    sub_1985800((__int64 *)(a1 + 32), (__int64 *)v17, &v33);
    v18 = v34.m128i_i64[0];
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = v16;
      *(__m128i *)(v17 + 8) = _mm_loadu_si128(&v34);
      *(_DWORD *)(v17 + 24) = v35;
      v17 = *(_QWORD *)(a1 + 40);
    }
    v18 = 0;
    *(_QWORD *)(a1 + 40) = v17 + 32;
  }
  _libc_free(v18);
  v12 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 5) - 1;
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
}
