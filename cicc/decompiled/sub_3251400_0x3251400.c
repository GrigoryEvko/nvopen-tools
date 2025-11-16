// Function: sub_3251400
// Address: 0x3251400
//
__int64 __fastcall sub_3251400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdi
  unsigned int v10; // esi
  __int64 v11; // r8
  int v12; // r11d
  __int64 v13; // rcx
  __int64 v14; // r15
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v17; // r10
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  __int64 result; // rax
  int v21; // eax
  int v22; // edx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rsi
  __int64 v26; // rcx
  char **v27; // rsi
  __int64 v28; // rdx
  char **v29; // rdi
  _BYTE *v30; // rdi
  int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  unsigned int v41; // r14d
  __int64 v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // [rsp+8h] [rbp-E8h]
  __int64 v45; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v46; // [rsp+68h] [rbp-88h]
  __int64 v47; // [rsp+70h] [rbp-80h]
  _BYTE v48[120]; // [rsp+78h] [rbp-78h] BYREF

  v9 = a1 + 344;
  v10 = *(_DWORD *)(a1 + 368);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 344);
    goto LABEL_29;
  }
  v11 = v10 - 1;
  v12 = 1;
  v13 = *(_QWORD *)(a1 + 352);
  v14 = 0;
  v15 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = v13 + 16LL * v15;
  v17 = *(_QWORD *)v16;
  if ( *(_QWORD *)v16 == a2 )
  {
LABEL_3:
    v18 = *(unsigned int *)(v16 + 8);
    goto LABEL_4;
  }
  while ( v17 != -4096 )
  {
    if ( !v14 && v17 == -8192 )
      v14 = v16;
    a6 = (unsigned int)(v12 + 1);
    v15 = v11 & (v12 + v15);
    v16 = v13 + 16LL * v15;
    v17 = *(_QWORD *)v16;
    if ( *(_QWORD *)v16 == a2 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v14 )
    v14 = v16;
  v21 = *(_DWORD *)(a1 + 360);
  ++*(_QWORD *)(a1 + 344);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v10 )
  {
LABEL_29:
    sub_A59910(v9, 2 * v10);
    v31 = *(_DWORD *)(a1 + 368);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 352);
      v34 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(a1 + 360) + 1;
      v14 = v33 + 16LL * v34;
      v35 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 != a2 )
      {
        a6 = 1;
        v11 = 0;
        while ( v35 != -4096 )
        {
          if ( !v11 && v35 == -8192 )
            v11 = v14;
          v34 = v32 & (a6 + v34);
          v14 = v33 + 16LL * v34;
          v35 = *(_QWORD *)v14;
          if ( *(_QWORD *)v14 == a2 )
            goto LABEL_17;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v11 )
          v14 = v11;
      }
      goto LABEL_17;
    }
    goto LABEL_56;
  }
  if ( v10 - *(_DWORD *)(a1 + 364) - v22 <= v10 >> 3 )
  {
    sub_A59910(v9, v10);
    v38 = *(_DWORD *)(a1 + 368);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 352);
      v11 = 1;
      v41 = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(a1 + 360) + 1;
      v42 = 0;
      v14 = v40 + 16LL * v41;
      v43 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 != a2 )
      {
        while ( v43 != -4096 )
        {
          if ( !v42 && v43 == -8192 )
            v42 = v14;
          a6 = (unsigned int)(v11 + 1);
          v41 = v39 & (v11 + v41);
          v14 = v40 + 16LL * v41;
          v43 = *(_QWORD *)v14;
          if ( *(_QWORD *)v14 == a2 )
            goto LABEL_17;
          v11 = (unsigned int)a6;
        }
        if ( v42 )
          v14 = v42;
      }
      goto LABEL_17;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 360);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 360) = v22;
  if ( *(_QWORD *)v14 != -4096 )
    --*(_DWORD *)(a1 + 364);
  *(_QWORD *)v14 = a2;
  *(_DWORD *)(v14 + 8) = 0;
  v23 = *(unsigned int *)(a1 + 384);
  v24 = *(unsigned int *)(a1 + 388);
  v25 = v23 + 1;
  v45 = a2;
  v47 = 0x800000000LL;
  v18 = v23;
  v46 = v48;
  if ( v23 + 1 > v24 )
  {
    v36 = *(_QWORD *)(a1 + 376);
    v37 = a1 + 376;
    if ( v36 > (unsigned __int64)&v45
      || (v24 = 5 * v23, v44 = *(_QWORD *)(a1 + 376), v23 = v36 + 88 * v23, (unsigned __int64)&v45 >= v23) )
    {
      sub_324BBF0(v37, v25, v23, v24, v11, a6);
      v23 = *(unsigned int *)(a1 + 384);
      v26 = *(_QWORD *)(a1 + 376);
      v27 = (char **)&v45;
      v18 = v23;
    }
    else
    {
      sub_324BBF0(v37, v25, v23, v24, v11, a6);
      v26 = *(_QWORD *)(a1 + 376);
      v23 = *(unsigned int *)(a1 + 384);
      v27 = (char **)((char *)&v45 + v26 - v44);
      v18 = v23;
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 376);
    v27 = (char **)&v45;
  }
  v28 = 11 * v23;
  v29 = (char **)(v26 + 8 * v28);
  if ( v29 )
  {
    *v29 = *v27;
    v29[1] = (char *)(v29 + 3);
    v29[2] = (char *)0x800000000LL;
    if ( *((_DWORD *)v27 + 4) )
      sub_32473B0((__int64)(v29 + 1), v27 + 1, v28, v26, v11, a6);
    v18 = *(unsigned int *)(a1 + 384);
  }
  v30 = v46;
  *(_DWORD *)(a1 + 384) = v18 + 1;
  if ( v30 != v48 )
  {
    _libc_free((unsigned __int64)v30);
    v18 = (unsigned int)(*(_DWORD *)(a1 + 384) - 1);
  }
  *(_DWORD *)(v14 + 8) = v18;
LABEL_4:
  v19 = *(_QWORD *)(a1 + 376) + 88 * v18;
  result = *(unsigned int *)(v19 + 16);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v19 + 20) )
  {
    sub_C8D5F0(v19 + 8, (const void *)(v19 + 24), result + 1, 8u, v11, a6);
    result = *(unsigned int *)(v19 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(v19 + 8) + 8 * result) = a3;
  ++*(_DWORD *)(v19 + 16);
  return result;
}
