// Function: sub_2B87810
// Address: 0x2b87810
//
__int64 __fastcall sub_2B87810(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  int v16; // r10d
  int v17; // edx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 v21; // rcx
  __int64 *v22; // rsi
  __int64 *v23; // rcx
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r12
  __int64 v28; // r14
  __int64 v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  int v32; // eax
  int v33; // ecx
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rsi
  unsigned __int64 v37; // r13
  __int64 v38; // rdi
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rsi
  unsigned int v42; // r14d
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+8h] [rbp-58h]
  unsigned __int64 v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v49; // [rsp+28h] [rbp-38h]

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_35;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = v10 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v8 == *(_QWORD *)v12 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v14 + 8;
  }
  v45 = 0;
  v16 = 1;
  while ( v13 != -4096 )
  {
    if ( !v45 )
    {
      if ( v13 != -8192 )
        v12 = 0;
      v45 = v12;
    }
    a6 = (unsigned int)(v16 + 1);
    v11 = (v9 - 1) & (v16 + v11);
    v12 = v10 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( v8 == *(_QWORD *)v12 )
      goto LABEL_3;
    ++v16;
  }
  if ( v45 )
    v12 = v45;
  ++*(_QWORD *)a1;
  v46 = v12;
  v17 = *(_DWORD *)(a1 + 16) + 1;
  if ( 4 * v17 >= 3 * v9 )
  {
LABEL_35:
    sub_B23080(a1, 2 * v9);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v35 = (v32 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v46 = v34 + 16LL * v35;
      v36 = *(_QWORD *)v46;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      if ( v8 != *(_QWORD *)v46 )
      {
        a6 = 1;
        v13 = 0;
        while ( v36 != -4096 )
        {
          if ( !v13 && v36 == -8192 )
            v13 = v46;
          v35 = v33 & (a6 + v35);
          v46 = v34 + 16LL * v35;
          v36 = *(_QWORD *)v46;
          if ( v8 == *(_QWORD *)v46 )
            goto LABEL_11;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( !v13 )
          v13 = v46;
        v46 = v13;
      }
      goto LABEL_11;
    }
    goto LABEL_68;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v17 <= v9 >> 3 )
  {
    sub_B23080(a1, v9);
    v39 = *(_DWORD *)(a1 + 24);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 8);
      v13 = 1;
      v42 = (v39 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v43 = 0;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v46 = v41 + 16LL * v42;
      v44 = *(_QWORD *)v46;
      if ( v8 != *(_QWORD *)v46 )
      {
        while ( v44 != -4096 )
        {
          if ( !v43 && v44 == -8192 )
            v43 = v46;
          a6 = (unsigned int)(v13 + 1);
          v42 = v40 & (v13 + v42);
          v46 = v41 + 16LL * v42;
          v44 = *(_QWORD *)v46;
          if ( v8 == *(_QWORD *)v46 )
            goto LABEL_11;
          v13 = (unsigned int)a6;
        }
        if ( !v43 )
          v43 = v46;
        v46 = v43;
      }
      goto LABEL_11;
    }
LABEL_68:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v46 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v49 = 0;
  *(_QWORD *)v46 = v8;
  *(_DWORD *)(v46 + 8) = 0;
  v18 = *(unsigned int *)(a1 + 44);
  v48 = *a2;
  v14 = *(unsigned int *)(a1 + 40);
  v19 = v14 + 1;
  v20 = v14;
  if ( v14 + 1 > v18 )
  {
    v37 = *(_QWORD *)(a1 + 32);
    v38 = a1 + 32;
    if ( v37 > (unsigned __int64)&v48 || (unsigned __int64)&v48 >= v37 + 16 * v14 )
    {
      sub_2B54CF0(v38, v19, v14, v18, v13, a6);
      v14 = *(unsigned int *)(a1 + 40);
      v21 = *(_QWORD *)(a1 + 32);
      v22 = &v48;
      v20 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2B54CF0(v38, v19, v14, v18, v13, a6);
      v21 = *(_QWORD *)(a1 + 32);
      v14 = *(unsigned int *)(a1 + 40);
      v22 = (__int64 *)((char *)&v48 + v21 - v37);
      v20 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 32);
    v22 = &v48;
  }
  v23 = (__int64 *)(16 * v14 + v21);
  if ( v23 )
  {
    *v23 = *v22;
    v23[1] = v22[1];
    v22[1] = 0;
    v20 = *(_DWORD *)(a1 + 40);
    v24 = v49;
    *(_DWORD *)(a1 + 40) = v20 + 1;
    v14 = v20;
    if ( v24 )
    {
      v25 = *(_QWORD *)(v24 + 144);
      if ( v25 != v24 + 160 )
        _libc_free(v25);
      sub_C7D6A0(*(_QWORD *)(v24 + 120), 8LL * *(unsigned int *)(v24 + 136), 8);
      sub_C7D6A0(*(_QWORD *)(v24 + 88), 16LL * *(unsigned int *)(v24 + 104), 8);
      v26 = *(_QWORD *)(v24 + 8);
      v27 = v26 + 8LL * *(unsigned int *)(v24 + 16);
      v47 = v26;
      while ( v47 != v27 )
      {
        v28 = *(_QWORD *)(v27 - 8);
        v27 -= 8LL;
        if ( v28 )
        {
          v29 = v28 + 160LL * *(_QWORD *)(v28 - 8);
          while ( v28 != v29 )
          {
            while ( 1 )
            {
              v29 -= 160;
              v30 = *(_QWORD *)(v29 + 88);
              if ( v30 != v29 + 104 )
                _libc_free(v30);
              v31 = *(_QWORD *)(v29 + 40);
              if ( v31 == v29 + 56 )
                break;
              _libc_free(v31);
              if ( v28 == v29 )
                goto LABEL_27;
            }
          }
LABEL_27:
          j_j_j___libc_free_0_0(v28 - 8);
        }
      }
      if ( v47 != v24 + 24 )
        _libc_free(v47);
      j_j___libc_free_0(v24);
      v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v20 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v20 + 1;
  }
  *(_DWORD *)(v46 + 8) = v20;
  return *(_QWORD *)(a1 + 32) + 16 * v14 + 8;
}
