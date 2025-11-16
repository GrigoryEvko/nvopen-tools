// Function: sub_2975CE0
// Address: 0x2975ce0
//
__int64 __fastcall sub_2975CE0(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // r12d
  char v9; // cl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // eax
  unsigned int *v13; // rdx
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  _DWORD *v18; // r14
  int v19; // edx
  unsigned int v20; // edi
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rcx
  char **v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdi
  _BYTE *v28; // rdi
  __int64 v29; // rcx
  int v30; // eax
  unsigned int v31; // edx
  int v32; // esi
  unsigned __int64 v33; // r15
  __int64 v34; // rdi
  __int64 v35; // rcx
  int v36; // edx
  unsigned int v37; // eax
  int v38; // esi
  _DWORD *v39; // rdi
  int v40; // eax
  int v41; // edx
  int v42; // [rsp+20h] [rbp-60h] BYREF
  _BYTE *v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  _BYTE v45[72]; // [rsp+38h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 3;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v11 = v16 - 1;
  }
  v12 = v11 & (37 * v8);
  v13 = (unsigned int *)(v10 + 8LL * v12);
  a5 = *v13;
  if ( v8 == (_DWORD)a5 )
  {
LABEL_4:
    v14 = v13[1];
    return *(_QWORD *)(a1 + 48) + 40 * v14 + 8;
  }
  a6 = 1;
  v18 = 0;
  while ( (_DWORD)a5 != -1 )
  {
    if ( !v18 && (_DWORD)a5 == -2 )
      v18 = v13;
    v12 = v11 & (a6 + v12);
    v13 = (unsigned int *)(v10 + 8LL * v12);
    a5 = *v13;
    if ( v8 == (_DWORD)a5 )
      goto LABEL_4;
    a6 = (unsigned int)(a6 + 1);
  }
  v17 = *(_DWORD *)(a1 + 8);
  if ( !v18 )
    v18 = v13;
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 12;
  v16 = 4;
LABEL_10:
  if ( 4 * v19 >= v20 )
  {
    sub_29758F0(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v29 = a1 + 16;
      v30 = 3;
    }
    else
    {
      v40 = *(_DWORD *)(a1 + 24);
      v29 = *(_QWORD *)(a1 + 16);
      if ( !v40 )
        goto LABEL_66;
      v30 = v40 - 1;
    }
    v31 = v30 & (37 * v8);
    v18 = (_DWORD *)(v29 + 8LL * v31);
    v32 = *v18;
    if ( v8 != *v18 )
    {
      a5 = 1;
      v39 = 0;
      while ( v32 != -1 )
      {
        if ( !v39 && v32 == -2 )
          v39 = v18;
        a6 = (unsigned int)(a5 + 1);
        v31 = v30 & (a5 + v31);
        v18 = (_DWORD *)(v29 + 8LL * v31);
        v32 = *v18;
        if ( v8 == *v18 )
          goto LABEL_32;
        a5 = (unsigned int)a6;
      }
      goto LABEL_41;
    }
LABEL_32:
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 12) - v19 <= v16 >> 3 )
  {
    sub_29758F0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = a1 + 16;
      v36 = 3;
      goto LABEL_38;
    }
    v41 = *(_DWORD *)(a1 + 24);
    v35 = *(_QWORD *)(a1 + 16);
    if ( v41 )
    {
      v36 = v41 - 1;
LABEL_38:
      v37 = v36 & (37 * v8);
      v18 = (_DWORD *)(v35 + 8LL * v37);
      v38 = *v18;
      if ( v8 != *v18 )
      {
        a5 = 1;
        v39 = 0;
        while ( v38 != -1 )
        {
          if ( !v39 && v38 == -2 )
            v39 = v18;
          a6 = (unsigned int)(a5 + 1);
          v37 = v36 & (a5 + v37);
          v18 = (_DWORD *)(v35 + 8LL * v37);
          v38 = *v18;
          if ( v8 == *v18 )
            goto LABEL_32;
          a5 = (unsigned int)a6;
        }
LABEL_41:
        if ( v39 )
          v18 = v39;
        goto LABEL_32;
      }
      goto LABEL_32;
    }
LABEL_66:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *v18 != -1 )
    --*(_DWORD *)(a1 + 12);
  *v18 = v8;
  v18[1] = 0;
  v21 = *(unsigned int *)(a1 + 60);
  v42 = *a2;
  v22 = *(unsigned int *)(a1 + 56);
  v23 = v22 + 1;
  v44 = 0x200000000LL;
  v14 = v22;
  v43 = v45;
  if ( v22 + 1 > v21 )
  {
    v33 = *(_QWORD *)(a1 + 48);
    v34 = a1 + 48;
    if ( v33 > (unsigned __int64)&v42 || (unsigned __int64)&v42 >= v33 + 40 * v22 )
    {
      sub_29757E0(v34, v23, v22, v21, a5, a6);
      v22 = *(unsigned int *)(a1 + 56);
      v24 = *(_QWORD *)(a1 + 48);
      v25 = (char **)&v42;
      v14 = v22;
    }
    else
    {
      sub_29757E0(v34, v23, v22, v21, a5, a6);
      v24 = *(_QWORD *)(a1 + 48);
      v22 = *(unsigned int *)(a1 + 56);
      v25 = (char **)((char *)&v42 + v24 - v33);
      v14 = v22;
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 48);
    v25 = (char **)&v42;
  }
  v26 = 5 * v22;
  v27 = v24 + 8 * v26;
  if ( v27 )
  {
    *(_DWORD *)v27 = *(_DWORD *)v25;
    *(_QWORD *)(v27 + 8) = v27 + 24;
    *(_QWORD *)(v27 + 16) = 0x200000000LL;
    if ( *((_DWORD *)v25 + 4) )
      sub_29739A0(v27 + 8, v25 + 1, v26, v24, a5, a6);
    v14 = *(unsigned int *)(a1 + 56);
  }
  v28 = v43;
  *(_DWORD *)(a1 + 56) = v14 + 1;
  if ( v28 != v45 )
  {
    _libc_free((unsigned __int64)v28);
    v14 = (unsigned int)(*(_DWORD *)(a1 + 56) - 1);
  }
  v18[1] = v14;
  return *(_QWORD *)(a1 + 48) + 40 * v14 + 8;
}
