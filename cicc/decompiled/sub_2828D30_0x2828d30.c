// Function: sub_2828D30
// Address: 0x2828d30
//
__int64 __fastcall sub_2828D30(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 v12; // r15
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  __int64 v23; // rcx
  char **v24; // rsi
  __int64 v25; // rdx
  char **v26; // rdi
  _BYTE *v27; // rdi
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rdi
  unsigned int v31; // eax
  __int64 v32; // rsi
  unsigned __int64 v33; // r14
  __int64 v34; // rdi
  int v35; // eax
  int v36; // eax
  __int64 v37; // rsi
  unsigned int v38; // r14d
  __int64 v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v42; // [rsp+58h] [rbp-88h]
  __int64 v43; // [rsp+60h] [rbp-80h]
  _BYTE v44[120]; // [rsp+68h] [rbp-78h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
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
    return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
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
LABEL_27:
    sub_D39D40(a1, 2 * v9);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = (v28 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v30 + 16LL * v31;
      v32 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v32 != -4096 )
        {
          if ( !v15 && v32 == -8192 )
            v15 = v12;
          v31 = v29 & (a6 + v31);
          v12 = v30 + 16LL * v31;
          v32 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
    goto LABEL_54;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_D39D40(a1, v9);
    v35 = *(_DWORD *)(a1 + 24);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v38 = v36 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v39 = 0;
      v12 = v37 + 16LL * v38;
      v40 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v40 != -4096 )
        {
          if ( !v39 && v40 == -8192 )
            v39 = v12;
          a6 = (unsigned int)(v15 + 1);
          v38 = v36 & (v15 + v38);
          v12 = v37 + 16LL * v38;
          v40 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = (unsigned int)a6;
        }
        if ( v39 )
          v12 = v39;
      }
      goto LABEL_15;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *(unsigned int *)(a1 + 44);
  v41 = *a2;
  v21 = *(unsigned int *)(a1 + 40);
  v22 = v21 + 1;
  v43 = 0x800000000LL;
  v16 = v21;
  v42 = v44;
  if ( v21 + 1 > v20 )
  {
    v33 = *(_QWORD *)(a1 + 32);
    v34 = a1 + 32;
    if ( v33 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v33 + 88 * v21 )
    {
      sub_23590C0(v34, v22, v21, v20, v15, a6);
      v21 = *(unsigned int *)(a1 + 40);
      v23 = *(_QWORD *)(a1 + 32);
      v24 = (char **)&v41;
      v16 = v21;
    }
    else
    {
      sub_23590C0(v34, v22, v21, v20, v15, a6);
      v23 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v24 = (char **)((char *)&v41 + v23 - v33);
      v16 = v21;
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 32);
    v24 = (char **)&v41;
  }
  v25 = 11 * v21;
  v26 = (char **)(v23 + 8 * v25);
  if ( v26 )
  {
    *v26 = *v24;
    v26[1] = (char *)(v26 + 3);
    v26[2] = (char *)0x800000000LL;
    if ( *((_DWORD *)v24 + 4) )
      sub_281DD40((__int64)(v26 + 1), v24 + 1, v25, v23, v15, a6);
    v16 = *(unsigned int *)(a1 + 40);
  }
  v27 = v42;
  *(_DWORD *)(a1 + 40) = v16 + 1;
  if ( v27 != v44 )
  {
    _libc_free((unsigned __int64)v27);
    v16 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
}
