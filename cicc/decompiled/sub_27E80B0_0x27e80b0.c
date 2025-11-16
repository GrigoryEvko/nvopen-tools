// Function: sub_27E80B0
// Address: 0x27e80b0
//
__int64 __fastcall sub_27E80B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 v12; // r14
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // r12
  _QWORD *v22; // rax
  int v23; // eax
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  unsigned int v31; // r15d
  __int64 v32; // rdi
  __int64 v33; // rcx

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v10 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_3:
    v16 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v16 + 8;
  }
  while ( v15 != -4096 )
  {
    if ( v15 == -8192 && !v12 )
      v12 = v14;
    a6 = (unsigned int)(v11 + 1);
    v13 = (v9 - 1) & (v11 + v13);
    v14 = v10 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_21:
    sub_B23080(a1, 2 * v9);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = (v23 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v25 + 16LL * v26;
      v27 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v27 != -4096 )
        {
          if ( !v15 && v27 == -8192 )
            v15 = v12;
          v26 = v24 & (a6 + v26);
          v12 = v25 + 16LL * v26;
          v27 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
    goto LABEL_44;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_B23080(a1, v9);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v31 = v29 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v32 = 0;
      v12 = v30 + 16LL * v31;
      v33 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v33 != -4096 )
        {
          if ( !v32 && v33 == -8192 )
            v32 = v12;
          a6 = (unsigned int)(v15 + 1);
          v31 = v29 & (v15 + v31);
          v12 = v30 + 16LL * v31;
          v33 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v32 )
          v12 = v32;
      }
      goto LABEL_15;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *(unsigned int *)(a1 + 40);
  v21 = *a2;
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v20 + 1, 0x10u, v15, a6);
    v20 = *(unsigned int *)(a1 + 40);
  }
  v22 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 16 * v20);
  *v22 = v21;
  v22[1] = 0;
  v16 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v16 + 1;
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 16 * v16 + 8;
}
