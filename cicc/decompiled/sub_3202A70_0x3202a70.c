// Function: sub_3202A70
// Address: 0x3202a70
//
__int64 __fastcall sub_3202A70(__int64 a1, __int64 *a2)
{
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v9; // r9
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  int v15; // eax
  int v16; // edx
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  __int64 v20; // rcx
  char *v21; // rsi
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  _BYTE *v24; // rdi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rsi
  unsigned __int64 v30; // r15
  __int64 v31; // rdi
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  unsigned int v35; // r15d
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  _BYTE v41[72]; // [rsp+48h] [rbp-48h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  LODWORD(v39) = 0;
  v38 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 & (37 * v4);
  v11 = v7 + 12LL * v10;
  v12 = *(_QWORD *)v11;
  if ( *(_QWORD *)v11 == v4 )
  {
LABEL_3:
    v13 = *(unsigned int *)(v11 + 8);
    return *(_QWORD *)(a1 + 32) + 40 * v13 + 8;
  }
  while ( v12 != -1 )
  {
    if ( !v8 && v12 == -2 )
      v8 = v11;
    v10 = v6 & (v9 + v10);
    v11 = v7 + 12LL * v10;
    v12 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 == v4 )
      goto LABEL_3;
    v9 = (unsigned int)(v9 + 1);
  }
  if ( !v8 )
    v8 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_27:
    sub_3202890(a1, 2 * v5);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = (v25 - 1) & (37 * v4);
      v8 = v27 + 12LL * v28;
      v29 = *(_QWORD *)v8;
      v16 = *(_DWORD *)(a1 + 16) + 1;
      if ( v4 != *(_QWORD *)v8 )
      {
        v9 = 1;
        v6 = 0;
        while ( v29 != -1 )
        {
          if ( !v6 && v29 == -2 )
            v6 = v8;
          v28 = v26 & (v9 + v28);
          v8 = v27 + 12LL * v28;
          v29 = *(_QWORD *)v8;
          if ( v4 == *(_QWORD *)v8 )
            goto LABEL_15;
          v9 = (unsigned int)(v9 + 1);
        }
        if ( v6 )
          v8 = v6;
      }
      goto LABEL_15;
    }
    goto LABEL_54;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_3202890(a1, v5);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v6 = 1;
      v35 = v33 & (37 * v4);
      v8 = v34 + 12LL * v35;
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v36 = 0;
      v37 = *(_QWORD *)v8;
      if ( *(_QWORD *)v8 != v4 )
      {
        while ( v37 != -1 )
        {
          if ( !v36 && v37 == -2 )
            v36 = v8;
          v9 = (unsigned int)(v6 + 1);
          v35 = v33 & (v6 + v35);
          v8 = v34 + 12LL * v35;
          v37 = *(_QWORD *)v8;
          if ( v4 == *(_QWORD *)v8 )
            goto LABEL_15;
          v6 = (unsigned int)v9;
        }
        if ( v36 )
          v8 = v36;
      }
      goto LABEL_15;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v4;
  *(_DWORD *)(v8 + 8) = 0;
  v17 = *(unsigned int *)(a1 + 44);
  v38 = *a2;
  v18 = *(unsigned int *)(a1 + 40);
  v19 = v18 + 1;
  v40 = 0x100000000LL;
  v13 = v18;
  v39 = v41;
  if ( v18 + 1 > v17 )
  {
    v30 = *(_QWORD *)(a1 + 32);
    v31 = a1 + 32;
    if ( v30 > (unsigned __int64)&v38 || (unsigned __int64)&v38 >= v30 + 40 * v18 )
    {
      sub_31FDC20(v31, v19, v18, v17, v6, v9);
      v18 = *(unsigned int *)(a1 + 40);
      v20 = *(_QWORD *)(a1 + 32);
      v21 = (char *)&v38;
      v13 = v18;
    }
    else
    {
      sub_31FDC20(v31, v19, v18, v17, v6, v9);
      v20 = *(_QWORD *)(a1 + 32);
      v18 = *(unsigned int *)(a1 + 40);
      v21 = (char *)&v38 + v20 - v30;
      v13 = v18;
    }
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 32);
    v21 = (char *)&v38;
  }
  v22 = 5 * v18;
  v23 = (_QWORD *)(v20 + 8 * v22);
  if ( v23 )
  {
    *v23 = *(_QWORD *)v21;
    v23[1] = v23 + 3;
    v23[2] = 0x100000000LL;
    if ( *((_DWORD *)v21 + 4) )
      sub_31F4130((__int64)(v23 + 1), (__int64)(v21 + 8), v22, v20, v6, v9);
    v13 = *(unsigned int *)(a1 + 40);
  }
  v24 = v39;
  *(_DWORD *)(a1 + 40) = v13 + 1;
  if ( v24 != v41 )
  {
    _libc_free((unsigned __int64)v24);
    v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v8 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 40 * v13 + 8;
}
