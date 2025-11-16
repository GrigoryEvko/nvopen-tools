// Function: sub_31711D0
// Address: 0x31711d0
//
__int64 __fastcall sub_31711D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  char v9; // cl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  __int64 v18; // r14
  int v19; // edx
  unsigned int v20; // edi
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rcx
  char **v25; // rsi
  __int64 v26; // rdx
  char **v27; // rdi
  _BYTE *v28; // rdi
  __int64 v29; // rcx
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rsi
  unsigned __int64 v33; // r15
  __int64 v34; // rdi
  __int64 v35; // rcx
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 v39; // rdi
  int v40; // edx
  int v41; // edx
  __int64 v42; // [rsp+20h] [rbp-60h] BYREF
  _BYTE *v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  _BYTE v45[72]; // [rsp+38h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 7;
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
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = v10 + 16LL * v12;
  a5 = *(_QWORD *)v13;
  if ( v8 == *(_QWORD *)v13 )
  {
LABEL_4:
    v14 = *(unsigned int *)(v13 + 8);
    return *(_QWORD *)(a1 + 144) + 40 * v14 + 8;
  }
  a6 = 1;
  v18 = 0;
  while ( a5 != -4096 )
  {
    if ( !v18 && a5 == -8192 )
      v18 = v13;
    v12 = v11 & (a6 + v12);
    v13 = v10 + 16LL * v12;
    a5 = *(_QWORD *)v13;
    if ( v8 == *(_QWORD *)v13 )
      goto LABEL_4;
    a6 = (unsigned int)(a6 + 1);
  }
  if ( !v18 )
    v18 = v13;
  v17 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 24;
  v16 = 8;
LABEL_10:
  if ( 4 * v19 >= v20 )
  {
    sub_24F62F0(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v29 = a1 + 16;
      v30 = 7;
    }
    else
    {
      v40 = *(_DWORD *)(a1 + 24);
      v29 = *(_QWORD *)(a1 + 16);
      if ( !v40 )
        goto LABEL_66;
      v30 = v40 - 1;
    }
    v31 = v30 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v18 = v29 + 16LL * v31;
    v32 = *(_QWORD *)v18;
    if ( v8 != *(_QWORD *)v18 )
    {
      a5 = 1;
      v39 = 0;
      while ( v32 != -4096 )
      {
        if ( !v39 && v32 == -8192 )
          v39 = v18;
        a6 = (unsigned int)(a5 + 1);
        v31 = v30 & (a5 + v31);
        v18 = v29 + 16LL * v31;
        v32 = *(_QWORD *)v18;
        if ( v8 == *(_QWORD *)v18 )
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
    sub_24F62F0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = a1 + 16;
      v36 = 7;
      goto LABEL_38;
    }
    v41 = *(_DWORD *)(a1 + 24);
    v35 = *(_QWORD *)(a1 + 16);
    if ( v41 )
    {
      v36 = v41 - 1;
LABEL_38:
      v37 = v36 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v18 = v35 + 16LL * v37;
      v38 = *(_QWORD *)v18;
      if ( v8 != *(_QWORD *)v18 )
      {
        a5 = 1;
        v39 = 0;
        while ( v38 != -4096 )
        {
          if ( !v39 && v38 == -8192 )
            v39 = v18;
          a6 = (unsigned int)(a5 + 1);
          v37 = v36 & (a5 + v37);
          v18 = v35 + 16LL * v37;
          v38 = *(_QWORD *)v18;
          if ( v8 == *(_QWORD *)v18 )
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
  if ( *(_QWORD *)v18 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v18 = v8;
  *(_DWORD *)(v18 + 8) = 0;
  v21 = *(unsigned int *)(a1 + 156);
  v42 = *a2;
  v22 = *(unsigned int *)(a1 + 152);
  v23 = v22 + 1;
  v44 = 0x200000000LL;
  v14 = v22;
  v43 = v45;
  if ( v22 + 1 > v21 )
  {
    v33 = *(_QWORD *)(a1 + 144);
    v34 = a1 + 144;
    if ( v33 > (unsigned __int64)&v42 || (unsigned __int64)&v42 >= v33 + 40 * v22 )
    {
      sub_24F5300(v34, v23, v22, v21, a5, a6);
      v22 = *(unsigned int *)(a1 + 152);
      v24 = *(_QWORD *)(a1 + 144);
      v25 = (char **)&v42;
      v14 = v22;
    }
    else
    {
      sub_24F5300(v34, v23, v22, v21, a5, a6);
      v24 = *(_QWORD *)(a1 + 144);
      v22 = *(unsigned int *)(a1 + 152);
      v25 = (char **)((char *)&v42 + v24 - v33);
      v14 = v22;
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 144);
    v25 = (char **)&v42;
  }
  v26 = 5 * v22;
  v27 = (char **)(v24 + 8 * v26);
  if ( v27 )
  {
    *v27 = *v25;
    v27[1] = (char *)(v27 + 3);
    v27[2] = (char *)0x200000000LL;
    if ( *((_DWORD *)v25 + 4) )
      sub_316FB10((__int64)(v27 + 1), v25 + 1, v26, v24, a5, a6);
    v14 = *(unsigned int *)(a1 + 152);
  }
  v28 = v43;
  *(_DWORD *)(a1 + 152) = v14 + 1;
  if ( v28 != v45 )
  {
    _libc_free((unsigned __int64)v28);
    v14 = (unsigned int)(*(_DWORD *)(a1 + 152) - 1);
  }
  *(_DWORD *)(v18 + 8) = v14;
  return *(_QWORD *)(a1 + 144) + 40 * v14 + 8;
}
