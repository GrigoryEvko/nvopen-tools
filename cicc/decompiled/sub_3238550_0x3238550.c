// Function: sub_3238550
// Address: 0x3238550
//
__int64 __fastcall sub_3238550(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r13
  unsigned int v10; // esi
  __int64 v11; // rcx
  int v12; // r11d
  __int64 v13; // r8
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r10
  int v18; // eax
  int v19; // edx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // rcx
  __int64 *v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  int v27; // ecx
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r10d
  __int64 v32; // r13
  __int64 v33; // r12
  _QWORD *v34; // rdx
  int v35; // eax
  int v36; // eax
  __int64 v37; // rsi
  unsigned int v38; // r15d
  __int64 v39; // rdi
  __int64 v40; // rcx

  v9 = *a2;
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = v11 + 16LL * v14;
  v16 = *(_QWORD *)v15;
  if ( v9 == *(_QWORD *)v15 )
    return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v15 + 8);
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v13 )
      v13 = v15;
    a6 = (unsigned int)(v12 + 1);
    v14 = (v10 - 1) & (v12 + v14);
    v15 = v11 + 16LL * v14;
    v16 = *(_QWORD *)v15;
    if ( v9 == *(_QWORD *)v15 )
      return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v15 + 8);
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v10 )
  {
LABEL_22:
    sub_A59910(a1, 2 * v10);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = (v26 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v13 = v28 + 16LL * v29;
      v30 = *(_QWORD *)v13;
      if ( v9 != *(_QWORD *)v13 )
      {
        v31 = 1;
        a6 = 0;
        while ( v30 != -4096 )
        {
          if ( !a6 && v30 == -8192 )
            a6 = v13;
          v29 = v27 & (v31 + v29);
          v13 = v28 + 16LL * v29;
          v30 = *(_QWORD *)v13;
          if ( v9 == *(_QWORD *)v13 )
            goto LABEL_14;
          ++v31;
        }
        if ( a6 )
          v13 = a6;
      }
      goto LABEL_14;
    }
    goto LABEL_48;
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v19 <= v10 >> 3 )
  {
    sub_A59910(a1, v10);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 8);
      a6 = 1;
      v38 = v36 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v39 = 0;
      v13 = v37 + 16LL * v38;
      v40 = *(_QWORD *)v13;
      if ( v9 != *(_QWORD *)v13 )
      {
        while ( v40 != -4096 )
        {
          if ( !v39 && v40 == -8192 )
            v39 = v13;
          v38 = v36 & (a6 + v38);
          v13 = v37 + 16LL * v38;
          v40 = *(_QWORD *)v13;
          if ( v9 == *(_QWORD *)v13 )
            goto LABEL_14;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v39 )
          v13 = v39;
      }
      goto LABEL_14;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v13 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)(v13 + 8) = 0;
  *(_QWORD *)v13 = v9;
  *(_DWORD *)(v13 + 8) = *(_DWORD *)(a1 + 40);
  v20 = *(unsigned int *)(a1 + 40);
  v21 = *(unsigned int *)(a1 + 44);
  v22 = *(_DWORD *)(a1 + 40);
  if ( v20 >= v21 )
  {
    v32 = *a3;
    v33 = *a2;
    if ( v21 < v20 + 1 )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v20 + 1, 0x10u, v20 + 1, a6);
      v20 = *(unsigned int *)(a1 + 40);
    }
    v34 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 16 * v20);
    *v34 = v33;
    v34[1] = v32;
    v23 = *(_QWORD *)(a1 + 32);
    v25 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v25;
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 32);
    v24 = (__int64 *)(v23 + 16 * v20);
    if ( v24 )
    {
      *v24 = *a2;
      v24[1] = *a3;
      v22 = *(_DWORD *)(a1 + 40);
      v23 = *(_QWORD *)(a1 + 32);
    }
    v25 = (unsigned int)(v22 + 1);
    *(_DWORD *)(a1 + 40) = v25;
  }
  return v23 + 16 * v25 - 16;
}
