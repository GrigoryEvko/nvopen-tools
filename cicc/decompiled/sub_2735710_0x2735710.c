// Function: sub_2735710
// Address: 0x2735710
//
__int64 __fastcall sub_2735710(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // ecx
  __int64 v22; // rax
  char *v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  _BYTE *v28; // r13
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // r15
  unsigned __int64 *v32; // r14
  int v33; // r8d
  unsigned int v34; // eax
  __int64 v35; // rdi
  int v36; // esi
  __int64 v37; // rcx
  unsigned __int64 v38; // r13
  __int64 v39; // rdi
  int v40; // edi
  int v41; // edi
  __int64 v42; // rsi
  __int64 v43; // r15
  __int64 v44; // rcx
  int v45; // eax
  __int64 v46; // [rsp+8h] [rbp-2A68h]
  __int64 v47; // [rsp+8h] [rbp-2A68h]
  __int64 v48; // [rsp+1520h] [rbp-1550h] BYREF
  _BYTE *v49; // [rsp+1528h] [rbp-1548h]
  __int64 v50; // [rsp+1530h] [rbp-1540h]
  _BYTE v51[5432]; // [rsp+1538h] [rbp-1538h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_33;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = v10 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v8 == *(_QWORD *)v12 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 8);
    return *(_QWORD *)(a1 + 32) + 5400 * v14 + 8;
  }
  v46 = 0;
  v16 = 1;
  while ( v13 != -4096 )
  {
    if ( !v46 )
    {
      if ( v13 != -8192 )
        v12 = 0;
      v46 = v12;
    }
    a6 = (unsigned int)(v16 + 1);
    v11 = (v9 - 1) & (v16 + v11);
    v12 = v10 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( v8 == *(_QWORD *)v12 )
      goto LABEL_3;
    ++v16;
  }
  if ( v46 )
    v12 = v46;
  ++*(_QWORD *)a1;
  v47 = v12;
  v17 = *(_DWORD *)(a1 + 16) + 1;
  if ( 4 * v17 >= 3 * v9 )
  {
LABEL_33:
    sub_2735530(a1, 2 * v9);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v13 = (unsigned int)(v33 - 1);
      a6 = *(_QWORD *)(a1 + 8);
      v34 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v35 = *(_QWORD *)(a6 + 16LL * v34);
      v47 = a6 + 16LL * v34;
      if ( v8 != v35 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( !v37 && v35 == -8192 )
            v37 = v47;
          v34 = v13 & (v36 + v34);
          v35 = *(_QWORD *)(a6 + 16LL * v34);
          v47 = a6 + 16LL * v34;
          if ( v8 == v35 )
            goto LABEL_11;
          ++v36;
        }
        if ( !v37 )
          v37 = v47;
        v47 = v37;
      }
      goto LABEL_11;
    }
    goto LABEL_66;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v17 <= v9 >> 3 )
  {
    sub_2735530(a1, v9);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v13 = *(_QWORD *)(a1 + 8);
      v42 = 0;
      LODWORD(v43) = v41 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v47 = v13 + 16LL * (unsigned int)v43;
      v44 = *(_QWORD *)v47;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v45 = 1;
      if ( v8 != *(_QWORD *)v47 )
      {
        while ( v44 != -4096 )
        {
          if ( !v42 && v44 == -8192 )
            v42 = v47;
          a6 = (unsigned int)(v45 + 1);
          v43 = v41 & (unsigned int)(v43 + v45);
          v44 = *(_QWORD *)(v13 + 16 * v43);
          v47 = v13 + 16 * v43;
          if ( v8 == v44 )
            goto LABEL_11;
          ++v45;
        }
        if ( !v42 )
          v42 = v47;
        v47 = v42;
      }
      goto LABEL_11;
    }
LABEL_66:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v47 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v47 = v8;
  *(_DWORD *)(v47 + 8) = 0;
  v48 = *a2;
  v18 = *(unsigned int *)(a1 + 40);
  v50 = 0x800000000LL;
  v19 = *(unsigned int *)(a1 + 44);
  v20 = v18 + 1;
  v49 = v51;
  v21 = v18;
  if ( v18 + 1 > v19 )
  {
    v38 = *(_QWORD *)(a1 + 32);
    v39 = a1 + 32;
    if ( v38 > (unsigned __int64)&v48 || (unsigned __int64)&v48 >= v38 + 5400 * v18 )
    {
      sub_2367990(v39, v20, v18, v18, v13, a6);
      v18 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v23 = (char *)&v48;
      v21 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2367990(v39, v20, v18, v18, v13, a6);
      v22 = *(_QWORD *)(a1 + 32);
      v18 = *(unsigned int *)(a1 + 40);
      v23 = (char *)&v48 + v22 - v38;
      v21 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = (char *)&v48;
  }
  v24 = (_QWORD *)(5400 * v18 + v22);
  if ( v24 )
  {
    v25 = *(_QWORD *)v23;
    v24[2] = 0x800000000LL;
    *v24 = v25;
    v24[1] = v24 + 3;
    v26 = *((unsigned int *)v23 + 4);
    if ( (_DWORD)v26 )
      sub_2732900((__int64)(v24 + 1), (__int64)(v23 + 8), v26, 0x800000000LL, v13, a6);
    v21 = *(_DWORD *)(a1 + 40);
  }
  v27 = (unsigned int)v50;
  v28 = v49;
  *(_DWORD *)(a1 + 40) = v21 + 1;
  v29 = (unsigned __int64)&v28[672 * v27];
  if ( v28 != (_BYTE *)v29 )
  {
    do
    {
      v30 = *(unsigned int *)(v29 - 648);
      v31 = *(_QWORD *)(v29 - 656);
      v29 -= 672LL;
      v30 *= 160;
      v32 = (unsigned __int64 *)(v31 + v30);
      if ( v31 != v31 + v30 )
      {
        do
        {
          v32 -= 20;
          if ( (unsigned __int64 *)*v32 != v32 + 2 )
            _libc_free(*v32);
        }
        while ( (unsigned __int64 *)v31 != v32 );
        v31 = *(_QWORD *)(v29 + 16);
      }
      if ( v31 != v29 + 32 )
        _libc_free(v31);
    }
    while ( v28 != (_BYTE *)v29 );
    v29 = (unsigned __int64)v49;
  }
  if ( (_BYTE *)v29 != v51 )
    _libc_free(v29);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v47 + 8) = v14;
  return *(_QWORD *)(a1 + 32) + 5400 * v14 + 8;
}
