// Function: sub_301A7F0
// Address: 0x301a7f0
//
__int64 __fastcall sub_301A7F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // rcx
  __int64 *v25; // rsi
  __int64 *v26; // rcx
  __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  unsigned __int64 v36; // r13
  __int64 v37; // rdi
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  unsigned int v41; // r15d
  __int64 v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+10h] [rbp-40h]
  __int64 v47; // [rsp+18h] [rbp-38h]

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_25;
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
    return *(_QWORD *)(a1 + 32) + 32 * v16 + 8;
  }
  while ( v15 != -4096 )
  {
    if ( !v12 && v15 == -8192 )
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
LABEL_25:
    sub_B23080(a1, 2 * v9);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = (v31 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v33 + 16LL * v34;
      v35 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v35 != -4096 )
        {
          if ( !v15 && v35 == -8192 )
            v15 = v12;
          v34 = v32 & (a6 + v34);
          v12 = v33 + 16LL * v34;
          v35 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
    goto LABEL_52;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_B23080(a1, v9);
    v38 = *(_DWORD *)(a1 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v41 = v39 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v42 = 0;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v40 + 16LL * v41;
      v43 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v43 != -4096 )
        {
          if ( !v42 && v43 == -8192 )
            v42 = v12;
          a6 = (unsigned int)(v15 + 1);
          v41 = v39 & (v15 + v41);
          v12 = v40 + 16LL * v41;
          v43 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v42 )
          v12 = v42;
      }
      goto LABEL_15;
    }
LABEL_52:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *a2;
  v21 = *(unsigned int *)(a1 + 44);
  v45 = 0;
  v44 = v20;
  v16 = *(unsigned int *)(a1 + 40);
  v46 = 0;
  v22 = v16 + 1;
  v47 = 0;
  v23 = v16;
  if ( v16 + 1 > v21 )
  {
    v36 = *(_QWORD *)(a1 + 32);
    v37 = a1 + 32;
    if ( v36 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v36 + 32 * v16 )
    {
      sub_3017590(v37, v22, v16, v21, v15, a6);
      v16 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v25 = &v44;
      v23 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_3017590(v37, v22, v16, v21, v15, a6);
      v24 = *(_QWORD *)(a1 + 32);
      v16 = *(unsigned int *)(a1 + 40);
      v25 = (__int64 *)((char *)&v44 + v24 - v36);
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = &v44;
  }
  v26 = (__int64 *)(32 * v16 + v24);
  if ( v26 )
  {
    *v26 = *v25;
    v27 = v25[1];
    v25[1] = 0;
    v28 = v45;
    v26[1] = v27;
    v29 = v25[2];
    v25[2] = 0;
    v26[2] = v29;
    v30 = v25[3];
    v25[3] = 0;
    v26[3] = v30;
    v23 = *(_DWORD *)(a1 + 40);
    *(_DWORD *)(a1 + 40) = v23 + 1;
    v16 = v23;
    if ( v28 )
    {
      j_j___libc_free_0(v28);
      v16 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v23 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v23 + 1;
  }
  *(_DWORD *)(v12 + 8) = v23;
  return *(_QWORD *)(a1 + 32) + 32 * v16 + 8;
}
