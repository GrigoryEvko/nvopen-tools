// Function: sub_2B58B10
// Address: 0x2b58b10
//
__int64 __fastcall sub_2B58B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rax
  unsigned __int8 **v18; // rbx
  unsigned __int8 **v19; // r14
  __int64 v20; // r9
  unsigned int v21; // r8d
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // rdi
  unsigned __int8 *v24; // r12
  __int64 v25; // r13
  unsigned int v26; // esi
  int v27; // eax
  int v28; // esi
  __int64 v29; // rdx
  unsigned int v30; // eax
  int v31; // edi
  unsigned __int8 **v32; // rcx
  unsigned __int8 *v33; // r8
  int v34; // r10d
  unsigned __int8 **v35; // r9
  int v37; // r11d
  int v38; // eax
  int v39; // eax
  int v40; // eax
  __int64 v41; // r8
  int v42; // r10d
  unsigned int v43; // edx
  unsigned __int8 *v44; // rsi
  __int64 v45; // r13
  unsigned __int64 v46; // rax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  unsigned int v50; // [rsp+8h] [rbp-48h]
  int v51; // [rsp+8h] [rbp-48h]
  unsigned __int64 v52[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(__int64 **)(a1 + 16);
  v8 = *v7;
  v9 = *v7 + ((unsigned __int64)*((unsigned int *)v7 + 2) << 6);
  while ( v8 != v9 )
  {
    while ( 1 )
    {
      v9 -= 64LL;
      if ( *(_BYTE *)(v9 + 28) )
        break;
      _libc_free(*(_QWORD *)(v9 + 8));
      if ( v8 == v9 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *((_DWORD *)v7 + 2) = 0;
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(unsigned int *)(v10 + 8);
  v12 = v11;
  if ( *(_DWORD *)(v10 + 12) <= (unsigned int)v11 )
  {
    v45 = sub_C8D7D0(*(_QWORD *)(a1 + 16), v10 + 16, 0, 0x40u, v52, a6);
    v46 = v45 + ((unsigned __int64)*(unsigned int *)(v10 + 8) << 6);
    if ( v46 )
    {
      *(_QWORD *)v46 = 0;
      *(_QWORD *)(v46 + 8) = v46 + 32;
      *(_QWORD *)(v46 + 16) = 4;
      *(_DWORD *)(v46 + 24) = 0;
      *(_BYTE *)(v46 + 28) = 1;
    }
    sub_2B44740(v10, v45);
    v47 = v52[0];
    if ( v10 + 16 != *(_QWORD *)v10 )
    {
      v51 = v52[0];
      _libc_free(*(_QWORD *)v10);
      v47 = v51;
    }
    *(_DWORD *)(v10 + 12) = v47;
    v48 = *(_DWORD *)(v10 + 8);
    *(_QWORD *)v10 = v45;
    v49 = (unsigned int)(v48 + 1);
    *(_DWORD *)(v10 + 8) = v49;
    v16 = v45 + (v49 << 6) - 64;
  }
  else
  {
    v13 = *(_QWORD *)v10;
    v14 = *(_QWORD *)v10 + (v11 << 6);
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = v14 + 32;
      *(_QWORD *)(v14 + 16) = 4;
      *(_DWORD *)(v14 + 24) = 0;
      *(_BYTE *)(v14 + 28) = 1;
      v12 = *(_DWORD *)(v10 + 8);
      v13 = *(_QWORD *)v10;
    }
    v15 = (unsigned int)(v12 + 1);
    *(_DWORD *)(v10 + 8) = v15;
    v16 = v13 + (v15 << 6) - 64;
  }
  if ( !*(_BYTE *)(v16 + 28) )
    goto LABEL_30;
  v17 = *(_QWORD **)(v16 + 8);
  v13 = *(unsigned int *)(v16 + 20);
  v14 = (__int64)&v17[v13];
  if ( v17 == (_QWORD *)v14 )
  {
LABEL_29:
    if ( (unsigned int)v13 >= *(_DWORD *)(v16 + 16) )
    {
LABEL_30:
      sub_C8CC70(v16, a2, v14, v13, a5, a6);
      goto LABEL_14;
    }
    *(_DWORD *)(v16 + 20) = v13 + 1;
    *(_QWORD *)v14 = a2;
    ++*(_QWORD *)v16;
  }
  else
  {
    while ( a2 != *v17 )
    {
      if ( (_QWORD *)v14 == ++v17 )
        goto LABEL_29;
    }
  }
LABEL_14:
  v18 = **(unsigned __int8 ****)a1;
  v19 = &v18[*(_QWORD *)(*(_QWORD *)a1 + 8LL)];
  if ( v18 != v19 )
  {
    while ( 1 )
    {
      v24 = *v18;
      if ( !(unsigned __int8)sub_2B0D8B0(*v18) )
      {
        v25 = *(_QWORD *)(a1 + 24);
        v26 = *(_DWORD *)(v25 + 24);
        if ( !v26 )
        {
          ++*(_QWORD *)v25;
          goto LABEL_21;
        }
        v20 = *(_QWORD *)(v25 + 8);
        v21 = (v26 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v22 = (unsigned __int8 **)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v24 != *v22 )
          break;
      }
LABEL_17:
      if ( v19 == ++v18 )
        return 1;
    }
    v37 = 1;
    v32 = 0;
    while ( v23 != (unsigned __int8 *)-4096LL )
    {
      if ( v32 || v23 != (unsigned __int8 *)-8192LL )
        v22 = v32;
      v21 = (v26 - 1) & (v37 + v21);
      v23 = *(unsigned __int8 **)(v20 + 16LL * v21);
      if ( v24 == v23 )
        goto LABEL_17;
      ++v37;
      v32 = v22;
      v22 = (unsigned __int8 **)(v20 + 16LL * v21);
    }
    if ( !v32 )
      v32 = v22;
    v38 = *(_DWORD *)(v25 + 16);
    ++*(_QWORD *)v25;
    v31 = v38 + 1;
    if ( 4 * (v38 + 1) >= 3 * v26 )
    {
LABEL_21:
      sub_CE2410(v25, 2 * v26);
      v27 = *(_DWORD *)(v25 + 24);
      if ( !v27 )
        goto LABEL_64;
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v25 + 8);
      v30 = (v27 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v31 = *(_DWORD *)(v25 + 16) + 1;
      v32 = (unsigned __int8 **)(v29 + 16LL * v30);
      v33 = *v32;
      if ( v24 == *v32 )
        goto LABEL_37;
      v34 = 1;
      v35 = 0;
      while ( v33 != (unsigned __int8 *)-4096LL )
      {
        if ( !v35 && v33 == (unsigned __int8 *)-8192LL )
          v35 = v32;
        v30 = v28 & (v34 + v30);
        v32 = (unsigned __int8 **)(v29 + 16LL * v30);
        v33 = *v32;
        if ( v24 == *v32 )
          goto LABEL_37;
        ++v34;
      }
    }
    else
    {
      if ( v26 - *(_DWORD *)(v25 + 20) - v31 > v26 >> 3 )
        goto LABEL_37;
      v50 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
      sub_CE2410(v25, v26);
      v39 = *(_DWORD *)(v25 + 24);
      if ( !v39 )
      {
LABEL_64:
        ++*(_DWORD *)(v25 + 16);
        BUG();
      }
      v40 = v39 - 1;
      v41 = *(_QWORD *)(v25 + 8);
      v42 = 1;
      v35 = 0;
      v43 = v40 & v50;
      v31 = *(_DWORD *)(v25 + 16) + 1;
      v32 = (unsigned __int8 **)(v41 + 16LL * (v40 & v50));
      v44 = *v32;
      if ( v24 == *v32 )
        goto LABEL_37;
      while ( v44 != (unsigned __int8 *)-4096LL )
      {
        if ( v44 == (unsigned __int8 *)-8192LL && !v35 )
          v35 = v32;
        v43 = v40 & (v42 + v43);
        v32 = (unsigned __int8 **)(v41 + 16LL * v43);
        v44 = *v32;
        if ( v24 == *v32 )
          goto LABEL_37;
        ++v42;
      }
    }
    if ( v35 )
      v32 = v35;
LABEL_37:
    *(_DWORD *)(v25 + 16) = v31;
    if ( *v32 != (unsigned __int8 *)-4096LL )
      --*(_DWORD *)(v25 + 20);
    *v32 = v24;
    *((_DWORD *)v32 + 2) = 0;
    goto LABEL_17;
  }
  return 1;
}
