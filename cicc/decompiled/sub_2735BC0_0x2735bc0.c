// Function: sub_2735BC0
// Address: 0x2735bc0
//
__int64 __fastcall sub_2735BC0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r11d
  __int64 v10; // r13
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rax
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rsi
  unsigned int v21; // r12d
  __int64 v22; // rcx
  __int64 *v23; // rsi
  __int64 *v24; // rcx
  __int64 v25; // rax
  unsigned __int64 *v26; // rdx
  __int64 v27; // rax
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // r14
  __int64 v30; // rax
  int v31; // eax
  int v32; // esi
  unsigned int v33; // eax
  __int64 v34; // rdi
  int v35; // r10d
  unsigned __int64 v36; // r12
  __int64 v37; // rdi
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdi
  unsigned int v41; // r14d
  __int64 v42; // rsi
  unsigned __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 *v45; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  v5 = *a2;
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  v11 = v7 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v12 = v8 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v5 == *(_QWORD *)v12 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 8);
    return *(_QWORD *)(a1 + 32) + 32 * v14 + 8;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = v7 & (v9 + v11);
    v12 = v8 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( v5 == *(_QWORD *)v12 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v16 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_29:
    sub_2735530(a1, 2 * v6);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      a3 = *(unsigned int *)(a1 + 16);
      v33 = (v31 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v17 = a3 + 1;
      v10 = v7 + 16LL * v33;
      v34 = *(_QWORD *)v10;
      if ( v5 != *(_QWORD *)v10 )
      {
        v35 = 1;
        v13 = 0;
        while ( v34 != -4096 )
        {
          if ( !v13 && v34 == -8192 )
            v13 = v10;
          v33 = v32 & (v35 + v33);
          v10 = v7 + 16LL * v33;
          v34 = *(_QWORD *)v10;
          if ( v5 == *(_QWORD *)v10 )
            goto LABEL_15;
          ++v35;
        }
        if ( v13 )
          v10 = v13;
      }
      goto LABEL_15;
    }
    goto LABEL_56;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v17 <= v6 >> 3 )
  {
    sub_2735530(a1, v6);
    v38 = *(_DWORD *)(a1 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 8);
      a3 = *(unsigned int *)(a1 + 16);
      v7 = 0;
      v41 = v39 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v13 = 1;
      v17 = a3 + 1;
      v10 = v40 + 16LL * v41;
      v42 = *(_QWORD *)v10;
      if ( v5 != *(_QWORD *)v10 )
      {
        while ( v42 != -4096 )
        {
          if ( !v7 && v42 == -8192 )
            v7 = v10;
          v41 = v39 & (v13 + v41);
          v10 = v40 + 16LL * v41;
          v42 = *(_QWORD *)v10;
          if ( v5 == *(_QWORD *)v10 )
            goto LABEL_15;
          v13 = (unsigned int)(v13 + 1);
        }
        if ( v7 )
          v10 = v7;
      }
      goto LABEL_15;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v10 = v5;
  *(_DWORD *)(v10 + 8) = 0;
  v18 = *a2;
  v19 = *(unsigned int *)(a1 + 44);
  v45 = 0;
  v44 = v18;
  v14 = *(unsigned int *)(a1 + 40);
  v46 = 0;
  v20 = v14 + 1;
  v47 = 0;
  v21 = v14;
  if ( v14 + 1 > v19 )
  {
    v36 = *(_QWORD *)(a1 + 32);
    v37 = a1 + 32;
    if ( v36 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v36 + 32 * v14 )
    {
      sub_2358E60(v37, v20, a3, v19, v7, v13);
      v14 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v23 = &v44;
      v21 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2358E60(v37, v20, a3, v19, v7, v13);
      v22 = *(_QWORD *)(a1 + 32);
      v14 = *(unsigned int *)(a1 + 40);
      v23 = (__int64 *)((char *)&v44 + v22 - v36);
      v21 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = &v44;
  }
  v24 = (__int64 *)(32 * v14 + v22);
  if ( v24 )
  {
    *v24 = *v23;
    v25 = v23[1];
    v23[1] = 0;
    v26 = v45;
    v24[1] = v25;
    v27 = v23[2];
    v23[2] = 0;
    v28 = v46;
    v29 = v26;
    v24[2] = v27;
    v30 = v23[3];
    v23[3] = 0;
    v24[3] = v30;
    v21 = *(_DWORD *)(a1 + 40);
    v43 = (unsigned __int64)v26;
    for ( *(_DWORD *)(a1 + 40) = v21 + 1; v28 != v29; v29 += 21 )
    {
      if ( (unsigned __int64 *)*v29 != v29 + 2 )
        _libc_free(*v29);
    }
    v14 = v21;
    if ( v43 )
    {
      j_j___libc_free_0(v43);
      v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v21 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v21 + 1;
  }
  *(_DWORD *)(v10 + 8) = v21;
  return *(_QWORD *)(a1 + 32) + 32 * v14 + 8;
}
