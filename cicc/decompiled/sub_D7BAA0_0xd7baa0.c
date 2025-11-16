// Function: sub_D7BAA0
// Address: 0xd7baa0
//
__int64 __fastcall sub_D7BAA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r15
  unsigned int v14; // edx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  int v19; // eax
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v23; // rax
  int v24; // eax
  int v25; // ecx
  unsigned int v26; // edi
  unsigned __int64 v27; // rax
  int v28; // r10d
  int v29; // eax
  int v30; // ecx
  __int64 v31; // rdi
  unsigned int v32; // esi
  unsigned __int64 v33; // rax

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = 0;
  v14 = v8 & 0xFFFFFFF8 & (v9 - 1);
  v15 = v10 + 16LL * v14;
  v16 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == v16 )
  {
LABEL_3:
    v17 = *(unsigned int *)(v15 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v17 + 8;
  }
  while ( v16 != -8 )
  {
    if ( v16 == -16 && !v13 )
      v13 = v15;
    a6 = (unsigned int)(v11 + 1);
    v14 = (v9 - 1) & (v11 + v14);
    v15 = v10 + 16LL * v14;
    v16 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 == v16 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v13 )
    v13 = v15;
  v19 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v9 )
  {
LABEL_21:
    sub_BB08F0(a1, 2 * v9);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      v26 = v8 & 0xFFFFFFF8 & (v24 - 1);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v13 = v16 + 16LL * v26;
      v27 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != v27 )
      {
        v28 = 1;
        a6 = 0;
        while ( v27 != -8 )
        {
          if ( !a6 && v27 == -16 )
            a6 = v13;
          v26 = v25 & (v28 + v26);
          v13 = v16 + 16LL * v26;
          v27 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == v27 )
            goto LABEL_15;
          ++v28;
        }
        if ( a6 )
          v13 = a6;
      }
      goto LABEL_15;
    }
    goto LABEL_44;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v20 <= v9 >> 3 )
  {
    sub_BB08F0(a1, v9);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      a6 = 1;
      v16 = 0;
      v32 = v12 & (v29 - 1);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v13 = v31 + 16LL * v32;
      v33 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v12 != v33 )
      {
        while ( v33 != -8 )
        {
          if ( !v16 && v33 == -16 )
            v16 = v13;
          v32 = v30 & (a6 + v32);
          v13 = v31 + 16LL * v32;
          v33 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v12 == v33 )
            goto LABEL_15;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v16 )
          v13 = v16;
      }
      goto LABEL_15;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v20;
  if ( (*(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v13 = v8;
  *(_DWORD *)(v13 + 8) = 0;
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *a2;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v21 + 1, 0x10u, v16, a6);
    v21 = *(unsigned int *)(a1 + 40);
  }
  v23 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 16 * v21);
  *v23 = v22;
  v23[1] = 0;
  v17 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v17 + 1;
  *(_DWORD *)(v13 + 8) = v17;
  return *(_QWORD *)(a1 + 32) + 16 * v17 + 8;
}
