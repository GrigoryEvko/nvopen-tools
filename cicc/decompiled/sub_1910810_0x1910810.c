// Function: sub_1910810
// Address: 0x1910810
//
unsigned __int64 __fastcall sub_1910810(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edi
  unsigned __int64 v12; // r15
  int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 v19; // rax
  int v20; // r9d
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rdx
  unsigned __int64 result; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdi
  unsigned int v28; // esi
  __int64 *v29; // rcx
  __int64 v30; // r9
  int v31; // r11d
  int v32; // edi
  int v33; // r8d
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rdi
  unsigned int v38; // esi
  __int64 *v39; // rcx
  __int64 v40; // r9
  int v41; // ecx
  int v42; // r10d
  int v43; // eax
  int v44; // ecx
  __int64 v45; // rdi
  unsigned int v46; // edx
  int v47; // esi
  int v48; // r10d
  unsigned __int64 v49; // r9
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rdi
  int v53; // r10d
  unsigned int v54; // edx
  int v55; // esi
  int v56; // ecx
  int v57; // r10d
  unsigned __int64 v58; // [rsp+0h] [rbp-40h]
  unsigned __int64 v59; // [rsp+8h] [rbp-38h]
  __int64 v60; // [rsp+8h] [rbp-38h]

  v4 = a1 + 376;
  v9 = *(_DWORD *)(a1 + 400);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 376);
    goto LABEL_38;
  }
  v10 = *(_QWORD *)(a1 + 384);
  v11 = (v9 - 1) & (37 * a2);
  v12 = v10 + 40LL * v11;
  v13 = *(_DWORD *)v12;
  if ( *(_DWORD *)v12 != a2 )
  {
    v31 = 1;
    result = 0;
    while ( v13 != -1 )
    {
      if ( v13 == -2 && !result )
        result = v12;
      v11 = (v9 - 1) & (v31 + v11);
      v12 = v10 + 40LL * v11;
      v13 = *(_DWORD *)v12;
      if ( *(_DWORD *)v12 == a2 )
        goto LABEL_3;
      ++v31;
    }
    v32 = *(_DWORD *)(a1 + 392);
    if ( !result )
      result = v12;
    ++*(_QWORD *)(a1 + 376);
    v33 = v32 + 1;
    if ( 4 * (v32 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 396) - v33 > v9 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(a1 + 392) = v33;
        if ( *(_DWORD *)result != -1 )
          --*(_DWORD *)(a1 + 396);
        *(_DWORD *)result = a2;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
        goto LABEL_27;
      }
      sub_190FC70(v4, v9);
      v50 = *(_DWORD *)(a1 + 400);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = *(_QWORD *)(a1 + 384);
        v49 = 0;
        v53 = 1;
        v54 = (v50 - 1) & (37 * a2);
        v33 = *(_DWORD *)(a1 + 392) + 1;
        result = v52 + 40LL * v54;
        v55 = *(_DWORD *)result;
        if ( *(_DWORD *)result == a2 )
          goto LABEL_24;
        while ( v55 != -1 )
        {
          if ( !v49 && v55 == -2 )
            v49 = result;
          v54 = v51 & (v53 + v54);
          result = v52 + 40LL * v54;
          v55 = *(_DWORD *)result;
          if ( *(_DWORD *)result == a2 )
            goto LABEL_24;
          ++v53;
        }
        goto LABEL_42;
      }
      goto LABEL_69;
    }
LABEL_38:
    sub_190FC70(v4, 2 * v9);
    v43 = *(_DWORD *)(a1 + 400);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 384);
      v46 = (v43 - 1) & (37 * a2);
      v33 = *(_DWORD *)(a1 + 392) + 1;
      result = v45 + 40LL * v46;
      v47 = *(_DWORD *)result;
      if ( *(_DWORD *)result == a2 )
        goto LABEL_24;
      v48 = 1;
      v49 = 0;
      while ( v47 != -1 )
      {
        if ( !v49 && v47 == -2 )
          v49 = result;
        v46 = v44 & (v48 + v46);
        result = v45 + 40LL * v46;
        v47 = *(_DWORD *)result;
        if ( *(_DWORD *)result == a2 )
          goto LABEL_24;
        ++v48;
      }
LABEL_42:
      if ( v49 )
        result = v49;
      goto LABEL_24;
    }
LABEL_69:
    ++*(_DWORD *)(a1 + 392);
    BUG();
  }
LABEL_3:
  if ( *(_QWORD *)(v12 + 8) )
  {
    v14 = *(_QWORD *)(a1 + 408);
    v15 = *(_QWORD *)(a1 + 416);
    *(_QWORD *)(a1 + 488) += 32LL;
    if ( ((v14 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v14 + 32 <= v15 - v14 )
    {
      result = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 408) = result + 32;
    }
    else
    {
      v16 = *(unsigned int *)(a1 + 432);
      v17 = 0x40000000000LL;
      v18 = *(_DWORD *)(a1 + 432) >> 7;
      if ( v18 < 0x1E )
        v17 = 4096LL << v18;
      v59 = v17;
      v19 = malloc(v17);
      v21 = v59;
      if ( !v19 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v16 = *(unsigned int *)(a1 + 432);
        v21 = v59;
        v19 = 0;
      }
      if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 436) )
      {
        v58 = v21;
        v60 = v19;
        sub_16CD150(a1 + 424, (const void *)(a1 + 440), 0, 8, v21, v20);
        v16 = *(unsigned int *)(a1 + 432);
        v21 = v58;
        v19 = v60;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 424) + 8 * v16) = v19;
      v22 = v19 + v21;
      result = (v19 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 416) = v22;
      ++*(_DWORD *)(a1 + 432);
      *(_QWORD *)(a1 + 408) = result + 32;
    }
    *(_QWORD *)result = a3;
    v24 = 0;
    *(_QWORD *)(result + 8) = a4;
    v25 = *(_QWORD *)(a1 + 24);
    v26 = *(unsigned int *)(v25 + 48);
    if ( !(_DWORD)v26 )
      goto LABEL_16;
    v27 = *(_QWORD *)(v25 + 32);
    v28 = (v26 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v29 = (__int64 *)(v27 + 16LL * v28);
    v30 = *v29;
    if ( a4 == *v29 )
    {
LABEL_14:
      if ( v29 != (__int64 *)(v27 + 16 * v26) )
      {
        v24 = v29[1];
LABEL_16:
        *(_QWORD *)(result + 24) = v24;
        *(_QWORD *)(result + 16) = *(_QWORD *)(v12 + 24);
        *(_QWORD *)(v12 + 24) = result;
        return result;
      }
    }
    else
    {
      v41 = 1;
      while ( v30 != -8 )
      {
        v42 = v41 + 1;
        v28 = (v26 - 1) & (v41 + v28);
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( a4 == *v29 )
          goto LABEL_14;
        v41 = v42;
      }
    }
    v24 = 0;
    goto LABEL_16;
  }
  result = v12;
LABEL_27:
  *(_QWORD *)(result + 8) = a3;
  v34 = 0;
  *(_QWORD *)(result + 16) = a4;
  v35 = *(_QWORD *)(a1 + 24);
  v36 = *(unsigned int *)(v35 + 48);
  if ( (_DWORD)v36 )
  {
    v37 = *(_QWORD *)(v35 + 32);
    v38 = (v36 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( a4 == *v39 )
    {
LABEL_29:
      if ( v39 != (__int64 *)(v37 + 16 * v36) )
      {
        v34 = v39[1];
        goto LABEL_31;
      }
    }
    else
    {
      v56 = 1;
      while ( v40 != -8 )
      {
        v57 = v56 + 1;
        v38 = (v36 - 1) & (v56 + v38);
        v39 = (__int64 *)(v37 + 16LL * v38);
        v40 = *v39;
        if ( a4 == *v39 )
          goto LABEL_29;
        v56 = v57;
      }
    }
    v34 = 0;
  }
LABEL_31:
  *(_QWORD *)(result + 32) = v34;
  return result;
}
