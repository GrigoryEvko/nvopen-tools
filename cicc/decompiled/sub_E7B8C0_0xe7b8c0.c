// Function: sub_E7B8C0
// Address: 0xe7b8c0
//
__int64 __fastcall sub_E7B8C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  unsigned __int64 v38; // r13
  __int64 v39; // rdi
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  unsigned int v43; // r15d
  __int64 v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // [rsp+0h] [rbp-50h] BYREF
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+10h] [rbp-40h]
  __int64 v49; // [rsp+18h] [rbp-38h]

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
    sub_E7B220(a1, 2 * v9);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 8);
      v36 = (v33 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v35 + 16LL * v36;
      v37 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v37 != -4096 )
        {
          if ( !v15 && v37 == -8192 )
            v15 = v12;
          v36 = v34 & (a6 + v36);
          v12 = v35 + 16LL * v36;
          v37 = *(_QWORD *)v12;
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
    sub_E7B220(a1, v9);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v43 = v41 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v44 = 0;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v42 + 16LL * v43;
      v45 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v45 != -4096 )
        {
          if ( !v44 && v45 == -8192 )
            v44 = v12;
          a6 = (unsigned int)(v15 + 1);
          v43 = v41 & (v15 + v43);
          v12 = v42 + 16LL * v43;
          v45 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v44 )
          v12 = v44;
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
  v47 = 0;
  v46 = v20;
  v16 = *(unsigned int *)(a1 + 40);
  v48 = 0;
  v22 = v16 + 1;
  v49 = 0;
  v23 = v16;
  if ( v16 + 1 > v21 )
  {
    v38 = *(_QWORD *)(a1 + 32);
    v39 = a1 + 32;
    if ( v38 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v38 + 32 * v16 )
    {
      sub_E79C70(v39, v22, v16, v21, v15, a6);
      v16 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v25 = &v46;
      v23 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_E79C70(v39, v22, v16, v21, v15, a6);
      v24 = *(_QWORD *)(a1 + 32);
      v16 = *(unsigned int *)(a1 + 40);
      v25 = (__int64 *)((char *)&v46 + v24 - v38);
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = &v46;
  }
  v26 = (__int64 *)(32 * v16 + v24);
  if ( v26 )
  {
    *v26 = *v25;
    v27 = v25[1];
    v25[1] = 0;
    v28 = v47;
    v26[1] = v27;
    v29 = v25[2];
    v25[2] = 0;
    v26[2] = v29;
    v30 = v25[3];
    v25[3] = 0;
    v31 = v49;
    v26[3] = v30;
    v23 = *(_DWORD *)(a1 + 40);
    v32 = v31 - v28;
    *(_DWORD *)(a1 + 40) = v23 + 1;
    v16 = v23;
    if ( v28 )
    {
      j_j___libc_free_0(v28, v32);
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
