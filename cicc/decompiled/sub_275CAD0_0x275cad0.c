// Function: sub_275CAD0
// Address: 0x275cad0
//
__int64 __fastcall sub_275CAD0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  char *v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // r8d
  __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  unsigned __int64 v38; // r12
  __int64 v39; // rdi
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  unsigned int v43; // r14d
  __int64 v44; // rdi
  __int64 v45; // rcx
  int v46; // [rsp+18h] [rbp-98h] BYREF
  __int64 v47; // [rsp+20h] [rbp-90h]
  int *v48; // [rsp+28h] [rbp-88h]
  int *v49; // [rsp+30h] [rbp-80h]
  __int64 v50; // [rsp+38h] [rbp-78h]
  _QWORD v51[2]; // [rsp+40h] [rbp-70h] BYREF
  int v52; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v53; // [rsp+58h] [rbp-58h]
  int *v54; // [rsp+60h] [rbp-50h]
  int *v55; // [rsp+68h] [rbp-48h]
  __int64 v56; // [rsp+70h] [rbp-40h]

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_26;
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
    return *(_QWORD *)(a1 + 32) + 56 * v16 + 8;
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
LABEL_26:
    sub_9BAAD0(a1, 2 * v9);
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
    goto LABEL_53;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    sub_9BAAD0(a1, v9);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v43 = v41 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v44 = 0;
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
LABEL_53:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v20 = *(unsigned int *)(a1 + 40);
  v48 = &v46;
  v21 = *(unsigned int *)(a1 + 44);
  v49 = &v46;
  v22 = *a2;
  v23 = v20 + 1;
  v54 = &v52;
  v51[0] = v22;
  v24 = (char *)v51;
  v55 = &v52;
  v25 = v20;
  v46 = 0;
  v47 = 0;
  v50 = 0;
  v52 = 0;
  v53 = 0;
  v56 = 0;
  if ( v20 + 1 > v21 )
  {
    v38 = *(_QWORD *)(a1 + 32);
    v39 = a1 + 32;
    if ( v38 > (unsigned __int64)v51 || (v25 = v38 + 56 * v20, (unsigned __int64)v51 >= v25) )
    {
      sub_2758850(v39, v23, v25, v20, v15, a6);
      v20 = *(unsigned int *)(a1 + 40);
      v26 = *(_QWORD *)(a1 + 32);
      v24 = (char *)v51;
      LODWORD(v25) = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2758850(v39, v23, v25, v20, v15, a6);
      v26 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v24 = (char *)v51 + v26 - v38;
      LODWORD(v25) = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 32);
  }
  v27 = v26 + 56 * v20;
  if ( v27 )
  {
    v28 = v27 + 16;
    *(_QWORD *)v27 = *(_QWORD *)v24;
    v29 = *((_QWORD *)v24 + 3);
    if ( v29 )
    {
      v30 = *((_DWORD *)v24 + 4);
      *(_QWORD *)(v27 + 24) = v29;
      *(_DWORD *)(v27 + 16) = v30;
      *(_QWORD *)(v27 + 32) = *((_QWORD *)v24 + 4);
      *(_QWORD *)(v27 + 40) = *((_QWORD *)v24 + 5);
      *(_QWORD *)(v29 + 8) = v28;
      v31 = *((_QWORD *)v24 + 6);
      *((_QWORD *)v24 + 3) = 0;
      *(_QWORD *)(v27 + 48) = v31;
      *((_QWORD *)v24 + 4) = v24 + 16;
      *((_QWORD *)v24 + 5) = v24 + 16;
      *((_QWORD *)v24 + 6) = 0;
    }
    else
    {
      *(_DWORD *)(v27 + 16) = 0;
      *(_QWORD *)(v27 + 24) = 0;
      *(_QWORD *)(v27 + 32) = v28;
      *(_QWORD *)(v27 + 40) = v28;
      *(_QWORD *)(v27 + 48) = 0;
    }
    LODWORD(v25) = *(_DWORD *)(a1 + 40);
  }
  v32 = v53;
  *(_DWORD *)(a1 + 40) = v25 + 1;
  sub_2754510(v32);
  sub_2754510(0);
  v16 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 56 * v16 + 8;
}
