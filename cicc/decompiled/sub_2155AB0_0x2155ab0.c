// Function: sub_2155AB0
// Address: 0x2155ab0
//
__int64 __fastcall sub_2155AB0(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // ecx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  unsigned int v20; // edx
  unsigned int v21; // edx
  __int64 v22; // r8
  __int64 v23; // rsi
  __int64 v24; // rcx
  _BYTE *v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r14
  int v29; // edx
  __int64 v30; // r10
  __int64 v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rax
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned int **v39; // rdi
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __int64 v42; // r14
  _BYTE *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rdx
  void *v48; // rdi
  size_t v49; // rdx
  _QWORD *v50; // rax
  __int64 v52; // [rsp+8h] [rbp-A8h]
  _BYTE *v54; // [rsp+18h] [rbp-98h]
  unsigned int v55; // [rsp+18h] [rbp-98h]
  unsigned int v56; // [rsp+24h] [rbp-8Ch] BYREF
  unsigned int *v57; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v58[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v59[2]; // [rsp+40h] [rbp-70h] BYREF
  unsigned int **v60; // [rsp+50h] [rbp-60h] BYREF
  _WORD *v61; // [rsp+58h] [rbp-58h]
  __int64 v62; // [rsp+60h] [rbp-50h]
  _WORD *v63; // [rsp+68h] [rbp-48h]
  int v64; // [rsp+70h] [rbp-40h]
  _QWORD *v65; // [rsp+78h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 48) )
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  v5 = *(_QWORD *)(a2 + 16);
  v6 = a2 + 8;
  v9 = a2 + 8;
  if ( v5 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v10 + 16);
        v12 = *(_QWORD *)(v10 + 24);
        if ( *(_DWORD *)(v10 + 32) <= a3 )
          break;
        v10 = *(_QWORD *)(v10 + 24);
        if ( !v12 )
          goto LABEL_9;
      }
      v9 = v10;
      v10 = *(_QWORD *)(v10 + 16);
    }
    while ( v11 );
  }
LABEL_9:
  v13 = *(_DWORD *)(v9 + 32);
  if ( v13 < a3 )
  {
    v20 = v13 + 1;
    LODWORD(v57) = v20;
    if ( v20 <= a3 )
    {
      while ( 1 )
      {
        v21 = v20 - 1;
        v22 = v6;
        LODWORD(v58[0]) = v21;
        if ( !v5 )
          goto LABEL_31;
        do
        {
          while ( 1 )
          {
            v23 = *(_QWORD *)(v5 + 16);
            v24 = *(_QWORD *)(v5 + 24);
            if ( v21 >= *(_DWORD *)(v5 + 32) )
              break;
            v5 = *(_QWORD *)(v5 + 24);
            if ( !v24 )
              goto LABEL_29;
          }
          v22 = v5;
          v5 = *(_QWORD *)(v5 + 16);
        }
        while ( v23 );
LABEL_29:
        if ( v6 == v22 || v21 > *(_DWORD *)(v22 + 32) )
        {
LABEL_31:
          v60 = (unsigned int **)v58;
          v22 = sub_21555E0((_QWORD *)a2, v22, (unsigned int **)&v60);
        }
        v25 = sub_21546A0(a2, (_BYTE *)(*(_QWORD *)(v22 + 40) + *(_QWORD *)(v22 + 48)));
        v26 = *(_QWORD *)(a2 + 16);
        v28 = v27;
        if ( !v26 )
          break;
        v29 = (int)v57;
        v30 = v6;
        do
        {
          while ( 1 )
          {
            v31 = *(_QWORD *)(v26 + 16);
            v32 = *(_QWORD *)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) <= (unsigned int)v57 )
              break;
            v26 = *(_QWORD *)(v26 + 24);
            if ( !v32 )
              goto LABEL_37;
          }
          v30 = v26;
          v26 = *(_QWORD *)(v26 + 16);
        }
        while ( v31 );
LABEL_37:
        if ( v6 == v30 || (unsigned int)v57 > *(_DWORD *)(v30 + 32) )
          goto LABEL_39;
LABEL_40:
        v20 = v29 + 1;
        *(_QWORD *)(v30 + 40) = v25;
        *(_QWORD *)(v30 + 48) = v28;
        *(_BYTE *)(v30 + 56) = 0;
        LODWORD(v57) = v20;
        if ( v20 > a3 )
          goto LABEL_10;
        v5 = *(_QWORD *)(a2 + 16);
      }
      v30 = v6;
LABEL_39:
      v54 = v25;
      v60 = &v57;
      v33 = sub_2155520((_QWORD *)a2, v30, (unsigned int **)&v60);
      v29 = (int)v57;
      v25 = v54;
      v30 = v33;
      goto LABEL_40;
    }
  }
LABEL_10:
  v14 = a3;
  if ( !a4 )
  {
    v15 = a3 - 1;
    if ( a3 - 1 < a3 - 2 || a3 == 1 )
    {
      v14 = a3;
    }
    else
    {
      while ( 1 )
      {
        v14 = v15 + 1;
        v16 = v6;
        if ( *(_QWORD *)(a2 + 16) )
        {
          v17 = *(_QWORD *)(a2 + 16);
          do
          {
            while ( 1 )
            {
              v18 = *(_QWORD *)(v17 + 16);
              v19 = *(_QWORD *)(v17 + 24);
              if ( *(_DWORD *)(v17 + 32) <= v15 )
                break;
              v17 = *(_QWORD *)(v17 + 24);
              if ( !v19 )
                goto LABEL_18;
            }
            v16 = v17;
            v17 = *(_QWORD *)(v17 + 16);
          }
          while ( v18 );
        }
LABEL_18:
        if ( *(_DWORD *)(v16 + 32) != v15 || *(_BYTE *)(v16 + 56) )
          break;
        if ( v15 == 1 || v15 - 1 < a3 - 2 )
        {
          v14 = v15;
          break;
        }
        --v15;
      }
    }
  }
  LOBYTE(v59[0]) = 0;
  v58[0] = v59;
  v55 = v14;
  v58[1] = 0;
  v60 = (unsigned int **)&unk_49EFBE0;
  v64 = 1;
  v63 = 0;
  v62 = 0;
  v61 = 0;
  v65 = v58;
  sub_16E7DE0((__int64)&v60, 10);
  v34 = v55;
  v56 = v55;
  while ( v34 <= a3 )
  {
    v35 = *(_QWORD *)(a2 + 16);
    v36 = v6;
    if ( !v35 )
      goto LABEL_51;
    do
    {
      while ( 1 )
      {
        v37 = *(_QWORD *)(v35 + 16);
        v38 = *(_QWORD *)(v35 + 24);
        if ( v34 >= *(_DWORD *)(v35 + 32) )
          break;
        v35 = *(_QWORD *)(v35 + 24);
        if ( !v38 )
          goto LABEL_49;
      }
      v36 = v35;
      v35 = *(_QWORD *)(v35 + 16);
    }
    while ( v37 );
LABEL_49:
    if ( v6 == v36 || v34 > *(_DWORD *)(v36 + 32) )
    {
LABEL_51:
      v57 = &v56;
      v36 = sub_2155520((_QWORD *)a2, v36, &v57);
    }
    *(_BYTE *)(v36 + 56) = 1;
    if ( (unsigned __int64)(v62 - (_QWORD)v63) <= 1 )
    {
      v39 = (unsigned int **)sub_16E7EE0((__int64)&v60, "//", 2u);
    }
    else
    {
      v39 = (unsigned int **)&v60;
      *v63++ = 12079;
    }
    v40 = sub_16E7EE0((__int64)v39, *(char **)(a2 + 64), *(_QWORD *)(a2 + 72));
    v41 = *(_BYTE **)(v40 + 24);
    if ( (unsigned __int64)v41 >= *(_QWORD *)(v40 + 16) )
    {
      v40 = sub_16E7DE0(v40, 58);
    }
    else
    {
      *(_QWORD *)(v40 + 24) = v41 + 1;
      *v41 = 58;
    }
    v42 = sub_16E7A90(v40, v56);
    v43 = *(_BYTE **)(v42 + 24);
    if ( (unsigned __int64)v43 >= *(_QWORD *)(v42 + 16) )
    {
      v42 = sub_16E7DE0(v42, 32);
      v44 = *(_QWORD *)(a2 + 16);
      if ( v44 )
      {
LABEL_58:
        v45 = v6;
        do
        {
          while ( 1 )
          {
            v46 = *(_QWORD *)(v44 + 16);
            v47 = *(_QWORD *)(v44 + 24);
            if ( *(_DWORD *)(v44 + 32) <= v56 )
              break;
            v44 = *(_QWORD *)(v44 + 24);
            if ( !v47 )
              goto LABEL_62;
          }
          v45 = v44;
          v44 = *(_QWORD *)(v44 + 16);
        }
        while ( v46 );
LABEL_62:
        if ( v6 != v45 && v56 <= *(_DWORD *)(v45 + 32) )
          goto LABEL_65;
        goto LABEL_64;
      }
    }
    else
    {
      *(_QWORD *)(v42 + 24) = v43 + 1;
      *v43 = 32;
      v44 = *(_QWORD *)(a2 + 16);
      if ( v44 )
        goto LABEL_58;
    }
    v45 = v6;
LABEL_64:
    v57 = &v56;
    v45 = sub_2155520((_QWORD *)a2, v45, &v57);
LABEL_65:
    v48 = *(void **)(v42 + 24);
    v49 = *(_QWORD *)(v45 + 48);
    if ( v49 > *(_QWORD *)(v42 + 16) - (_QWORD)v48 )
    {
      sub_16E7EE0(v42, *(char **)(v45 + 40), v49);
    }
    else if ( v49 )
    {
      v52 = *(_QWORD *)(v45 + 48);
      memcpy(v48, *(const void **)(v45 + 40), v49);
      *(_QWORD *)(v42 + 24) += v52;
    }
    v34 = v56 + 1;
    v56 = v34;
  }
  if ( v63 != v61 )
    sub_16E7BA0((__int64 *)&v60);
  v50 = v65;
  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)a1 = a1 + 16;
  sub_214ADD0((__int64 *)a1, (_BYTE *)*v50, *v50 + v50[1]);
  sub_16E7BC0((__int64 *)&v60);
  if ( (_QWORD *)v58[0] != v59 )
    j_j___libc_free_0(v58[0], v59[0] + 1LL);
  return a1;
}
