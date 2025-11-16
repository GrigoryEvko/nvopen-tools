// Function: sub_E9F050
// Address: 0xe9f050
//
__int64 __fastcall sub_E9F050(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  unsigned int v24; // edi
  __int64 v25; // rcx
  char *v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rsi
  int v35; // eax
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rsi
  unsigned __int64 v39; // r13
  __int64 v40; // rdi
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  unsigned int v44; // r15d
  __int64 v45; // rdi
  __int64 v46; // [rsp+0h] [rbp-60h] BYREF
  __int64 v47; // [rsp+8h] [rbp-58h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+18h] [rbp-48h]
  int v50; // [rsp+20h] [rbp-40h]
  __int64 v51; // [rsp+28h] [rbp-38h]

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
    return *(_QWORD *)(a1 + 32) + 48 * v16 + 8;
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
    sub_E9EE70(a1, 2 * v9);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v20 = (unsigned int)(v35 - 1);
      v36 = *(_QWORD *)(a1 + 8);
      v37 = v20 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v36 + 16LL * v37;
      v38 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v38 != -4096 )
        {
          if ( !v15 && v38 == -8192 )
            v15 = v12;
          v37 = v20 & (a6 + v37);
          v12 = v36 + 16LL * v37;
          v38 = *(_QWORD *)v12;
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
  v20 = v9 >> 3;
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= (unsigned int)v20 )
  {
    sub_E9EE70(a1, v9);
    v41 = *(_DWORD *)(a1 + 24);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v44 = v42 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v45 = 0;
      v12 = v43 + 16LL * v44;
      v20 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v20 != -4096 )
        {
          if ( !v45 && v20 == -8192 )
            v45 = v12;
          a6 = (unsigned int)(v15 + 1);
          v44 = v42 & (v15 + v44);
          v12 = v43 + 16LL * v44;
          v20 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v45 )
          v12 = v45;
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
  v21 = *a2;
  v22 = *(unsigned int *)(a1 + 44);
  v47 = 0;
  v46 = v21;
  v16 = *(unsigned int *)(a1 + 40);
  v48 = 0;
  v23 = v16 + 1;
  v49 = 0;
  v24 = v16;
  v50 = 0;
  v51 = 0;
  if ( v16 + 1 > v22 )
  {
    v39 = *(_QWORD *)(a1 + 32);
    v40 = a1 + 32;
    if ( v39 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v39 + 48 * v16 )
    {
      sub_E9E190(v40, v23, v22, v20, v15, a6);
      v16 = *(unsigned int *)(a1 + 40);
      v25 = *(_QWORD *)(a1 + 32);
      v26 = (char *)&v46;
      v24 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_E9E190(v40, v23, v22, v20, v15, a6);
      v25 = *(_QWORD *)(a1 + 32);
      v16 = *(unsigned int *)(a1 + 40);
      v26 = (char *)&v46 + v25 - v39;
      v24 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = (char *)&v46;
  }
  v27 = v25 + 48 * v16;
  if ( v27 )
  {
    *(_QWORD *)v27 = *(_QWORD *)v26;
    v28 = *((_QWORD *)v26 + 1);
    *((_QWORD *)v26 + 1) = 0;
    v29 = v47;
    *(_QWORD *)(v27 + 8) = v28;
    v30 = *((_QWORD *)v26 + 2);
    *((_QWORD *)v26 + 2) = 0;
    *(_QWORD *)(v27 + 16) = v30;
    v31 = *((_QWORD *)v26 + 3);
    *((_QWORD *)v26 + 3) = 0;
    *(_QWORD *)(v27 + 24) = v31;
    *(_DWORD *)(v27 + 32) = *((_DWORD *)v26 + 8);
    v32 = *((_QWORD *)v26 + 5);
    v33 = v49;
    *(_QWORD *)(v27 + 40) = v32;
    v24 = *(_DWORD *)(a1 + 40);
    v34 = v33 - v29;
    *(_DWORD *)(a1 + 40) = v24 + 1;
    v16 = v24;
    if ( v29 )
    {
      j_j___libc_free_0(v29, v34);
      v16 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v24 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v24 + 1;
  }
  *(_DWORD *)(v12 + 8) = v24;
  return *(_QWORD *)(a1 + 32) + 48 * v16 + 8;
}
