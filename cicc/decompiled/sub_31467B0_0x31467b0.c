// Function: sub_31467B0
// Address: 0x31467b0
//
__int64 __fastcall sub_31467B0(__int64 a1, unsigned int *a2, __int64 *a3)
{
  int v6; // r13d
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // r9
  _DWORD *v10; // rdx
  int v11; // r10d
  unsigned int v12; // edi
  _DWORD *v13; // rax
  int v14; // ecx
  int v16; // eax
  int v17; // ecx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // eax
  int v28; // edi
  int v29; // r10d
  __int64 v30; // r13
  __int64 v31; // r12
  _QWORD *v32; // rdx
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  unsigned int v36; // r15d
  _DWORD *v37; // r8
  int v38; // esi

  v6 = *a2;
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v7 - 1;
  v10 = 0;
  v11 = 1;
  v12 = v9 & (37 * v6);
  v13 = (_DWORD *)(v8 + 8LL * v12);
  v14 = *v13;
  if ( v6 == *v13 )
    return *(_QWORD *)(a1 + 32) + 16LL * (unsigned int)v13[1];
  while ( v14 != -1 )
  {
    if ( v14 == -2 && !v10 )
      v10 = v13;
    v12 = v9 & (v11 + v12);
    v13 = (_DWORD *)(v8 + 8LL * v12);
    v14 = *v13;
    if ( v6 == *v13 )
      return *(_QWORD *)(a1 + 32) + 16LL * (unsigned int)v13[1];
    ++v11;
  }
  if ( !v10 )
    v10 = v13;
  v16 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_22:
    sub_A09770(a1, 2 * v7);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = (v24 - 1) & (37 * v6);
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_DWORD *)(v26 + 8LL * v27);
      v28 = *v10;
      if ( v6 != *v10 )
      {
        v29 = 1;
        v9 = 0;
        while ( v28 != -1 )
        {
          if ( !v9 && v28 == -2 )
            v9 = (__int64)v10;
          v27 = v25 & (v29 + v27);
          v10 = (_DWORD *)(v26 + 8LL * v27);
          v28 = *v10;
          if ( v6 == *v10 )
            goto LABEL_14;
          ++v29;
        }
        if ( v9 )
          v10 = (_DWORD *)v9;
      }
      goto LABEL_14;
    }
    goto LABEL_48;
  }
  if ( v7 - *(_DWORD *)(a1 + 20) - v17 <= v7 >> 3 )
  {
    sub_A09770(a1, v7);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 8);
      v9 = 1;
      v36 = v34 & (37 * v6);
      v37 = 0;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_DWORD *)(v35 + 8LL * v36);
      v38 = *v10;
      if ( v6 != *v10 )
      {
        while ( v38 != -1 )
        {
          if ( !v37 && v38 == -2 )
            v37 = v10;
          v36 = v34 & (v9 + v36);
          v10 = (_DWORD *)(v35 + 8LL * v36);
          v38 = *v10;
          if ( v6 == *v10 )
            goto LABEL_14;
          v9 = (unsigned int)(v9 + 1);
        }
        if ( v37 )
          v10 = v37;
      }
      goto LABEL_14;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *v10 != -1 )
    --*(_DWORD *)(a1 + 20);
  v10[1] = 0;
  *v10 = v6;
  v10[1] = *(_DWORD *)(a1 + 40);
  v18 = *(unsigned int *)(a1 + 40);
  v19 = *(unsigned int *)(a1 + 44);
  v20 = *(_DWORD *)(a1 + 40);
  if ( v18 >= v19 )
  {
    v30 = *a3;
    v31 = *a2;
    if ( v19 < v18 + 1 )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v18 + 1, 0x10u, v18 + 1, v9);
      v18 = *(unsigned int *)(a1 + 40);
    }
    v32 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 16 * v18);
    *v32 = v31;
    v32[1] = v30;
    v21 = *(_QWORD *)(a1 + 32);
    v23 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v23;
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 32);
    v22 = v21 + 16 * v18;
    if ( v22 )
    {
      *(_DWORD *)v22 = *a2;
      *(_QWORD *)(v22 + 8) = *a3;
      v20 = *(_DWORD *)(a1 + 40);
      v21 = *(_QWORD *)(a1 + 32);
    }
    v23 = (unsigned int)(v20 + 1);
    *(_DWORD *)(a1 + 40) = v23;
  }
  return v21 + 16 * v23 - 16;
}
