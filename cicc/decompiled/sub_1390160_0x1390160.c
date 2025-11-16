// Function: sub_1390160
// Address: 0x1390160
//
__int64 __fastcall sub_1390160(__int64 *a1, unsigned int a2, unsigned int a3)
{
  __int64 v4; // r14
  unsigned int v6; // edx
  unsigned int v7; // ecx
  int *v8; // r8
  int v9; // edi
  __int64 v10; // r12
  unsigned int v11; // r10d
  unsigned int *v12; // r9
  unsigned int v13; // edi
  __int64 result; // rax
  __int64 v15; // r13
  unsigned int v16; // esi
  __int64 v17; // rax
  int v18; // edx
  int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rsi
  unsigned int *v22; // rcx
  unsigned int v23; // edi
  int v24; // eax
  int v25; // r11d
  unsigned int *v26; // r9
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 *v29; // rdx
  int v30; // r8d
  int v31; // r9d
  int v32; // r11d
  int v33; // eax
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // edx
  int v38; // edx
  __int64 v39; // rdi
  int v40; // r11d
  __int64 v41; // r8
  unsigned int v42; // esi
  __int64 v43; // [rsp+8h] [rbp-68h]
  int v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  __m128i v47; // [rsp+20h] [rbp-50h] BYREF
  __int64 v48; // [rsp+30h] [rbp-40h]

  v4 = 0;
  v45 = a2;
  while ( 1 )
  {
    v15 = *a1;
    v16 = *(_DWORD *)(*a1 + 24);
    v17 = *(_QWORD *)(*a1 + 8);
    if ( !v16 )
    {
      v10 = *(_QWORD *)(a1[1] + 32) + 16LL * a3;
      ++*(_QWORD *)v15;
      goto LABEL_10;
    }
    v6 = v16 - 1;
    v7 = (v16 - 1) & (37 * a3);
    v8 = (int *)(v17 + 12LL * v7);
    v9 = *v8;
    if ( *v8 == a3 )
      break;
    v30 = 1;
    while ( v9 != -1 )
    {
      v31 = v30 + 1;
      v7 = v6 & (v30 + v7);
      v8 = (int *)(v17 + 12LL * v7);
      v9 = *v8;
      if ( *v8 == a3 )
        goto LABEL_3;
      v30 = v31;
    }
LABEL_4:
    v10 = *(_QWORD *)(a1[1] + 32) + 16LL * a3;
    v11 = v6 & (37 * a3);
    v12 = (unsigned int *)(v17 + 12LL * v11);
    v13 = *v12;
    if ( *v12 == a3 )
      goto LABEL_5;
    v32 = 1;
    v22 = 0;
    while ( v13 != -1 )
    {
      if ( v22 || v13 != -2 )
        v12 = v22;
      v11 = v6 & (v32 + v11);
      v13 = *(_DWORD *)(v17 + 12LL * v11);
      if ( v13 == a3 )
        goto LABEL_5;
      ++v32;
      v22 = v12;
      v12 = (unsigned int *)(v17 + 12LL * v11);
    }
    v33 = *(_DWORD *)(v15 + 16);
    if ( !v22 )
      v22 = v12;
    ++*(_QWORD *)v15;
    v24 = v33 + 1;
    if ( 4 * v24 < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(v15 + 20) - v24 > v16 >> 3 )
        goto LABEL_31;
      v44 = 37 * a3;
      sub_138FFA0(v15, v16);
      v37 = *(_DWORD *)(v15 + 24);
      if ( !v37 )
      {
LABEL_57:
        ++*(_DWORD *)(v15 + 16);
        BUG();
      }
      v38 = v37 - 1;
      v39 = *(_QWORD *)(v15 + 8);
      v26 = 0;
      v40 = 1;
      LODWORD(v41) = v38 & v44;
      v22 = (unsigned int *)(v39 + 12LL * (v38 & (unsigned int)v44));
      v42 = *v22;
      v24 = *(_DWORD *)(v15 + 16) + 1;
      if ( *v22 == a3 )
        goto LABEL_31;
      while ( v42 != -1 )
      {
        if ( v42 == -2 && !v26 )
          v26 = v22;
        v41 = v38 & (unsigned int)(v41 + v40);
        v22 = (unsigned int *)(v39 + 12 * v41);
        v42 = *v22;
        if ( *v22 == a3 )
          goto LABEL_31;
        ++v40;
      }
      goto LABEL_14;
    }
LABEL_10:
    sub_138FFA0(v15, 2 * v16);
    v18 = *(_DWORD *)(v15 + 24);
    if ( !v18 )
      goto LABEL_57;
    v19 = v18 - 1;
    v20 = *(_QWORD *)(v15 + 8);
    LODWORD(v21) = v19 & (37 * a3);
    v22 = (unsigned int *)(v20 + 12LL * (unsigned int)v21);
    v23 = *v22;
    v24 = *(_DWORD *)(v15 + 16) + 1;
    if ( *v22 == a3 )
      goto LABEL_31;
    v25 = 1;
    v26 = 0;
    while ( v23 != -1 )
    {
      if ( !v26 && v23 == -2 )
        v26 = v22;
      v21 = v19 & (unsigned int)(v21 + v25);
      v22 = (unsigned int *)(v20 + 12 * v21);
      v23 = *v22;
      if ( *v22 == a3 )
        goto LABEL_31;
      ++v25;
    }
LABEL_14:
    if ( v26 )
      v22 = v26;
LABEL_31:
    *(_DWORD *)(v15 + 16) = v24;
    if ( *v22 != -1 )
      --*(_DWORD *)(v15 + 20);
    *v22 = a3;
    v22[2] = v4;
    v22[1] = a2;
LABEL_5:
    result = sub_14C8220(*(_QWORD *)(v10 + 8));
    if ( result )
    {
      v27 = a1[1];
      v28 = *(unsigned int *)(v27 + 272);
      if ( (unsigned int)v28 >= *(_DWORD *)(v27 + 276) )
      {
        v43 = result;
        sub_16CD150(v27 + 264, v27 + 280, 0, 16);
        v28 = *(unsigned int *)(v27 + 272);
        result = v43;
      }
      v29 = (__int64 *)(*(_QWORD *)(v27 + 264) + 16 * v28);
      *v29 = v45 | (v4 << 32);
      v29[1] = result;
      ++*(_DWORD *)(v27 + 272);
      a3 = *(_DWORD *)(v10 + 4);
      if ( a3 == -1 )
        return result;
    }
    else
    {
      a3 = *(_DWORD *)(v10 + 4);
      if ( a3 == -1 )
        return result;
    }
    v4 = (unsigned int)(v4 + 1);
  }
LABEL_3:
  if ( v8 == (int *)(v17 + 12LL * v16) )
    goto LABEL_4;
  if ( v8[2] != (_DWORD)v4 || (result = a2, v8[1] != a2) )
  {
    v34 = a1[1];
    v47.m128i_i64[0] = __PAIR64__(v4, a2);
    v47.m128i_i64[1] = *(_QWORD *)(v8 + 1);
    v48 = 0x7FFFFFFFFFFFFFFFLL;
    v35 = *(unsigned int *)(v34 + 64);
    if ( (unsigned int)v35 >= *(_DWORD *)(v34 + 68) )
    {
      sub_16CD150(v34 + 56, v34 + 72, 0, 24);
      v35 = *(unsigned int *)(v34 + 64);
    }
    result = *(_QWORD *)(v34 + 56) + 24 * v35;
    v36 = v48;
    *(__m128i *)result = _mm_load_si128(&v47);
    *(_QWORD *)(result + 16) = v36;
    ++*(_DWORD *)(v34 + 64);
  }
  return result;
}
