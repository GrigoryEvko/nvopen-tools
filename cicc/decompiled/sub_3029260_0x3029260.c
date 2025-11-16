// Function: sub_3029260
// Address: 0x3029260
//
__int64 __fastcall sub_3029260(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned int v13; // edx
  unsigned int v14; // r8d
  unsigned int v15; // r10d
  unsigned int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned int v21; // edx
  unsigned int v22; // edx
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rcx
  _BYTE *v26; // r9
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  int v30; // edx
  __int64 v31; // r10
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rax
  _BYTE *v35; // rax
  unsigned int v36; // r8d
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rdx
  unsigned int **v41; // rdi
  __int64 v42; // rdi
  _BYTE *v43; // rax
  __int64 v44; // r14
  _BYTE *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // rdx
  void *v50; // rdi
  size_t v51; // rdx
  unsigned __int64 *v52; // rax
  __int64 v54; // [rsp+8h] [rbp-B8h]
  _BYTE *v56; // [rsp+18h] [rbp-A8h]
  unsigned int v57; // [rsp+18h] [rbp-A8h]
  unsigned int v58; // [rsp+24h] [rbp-9Ch] BYREF
  unsigned int *v59; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v60[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v61[16]; // [rsp+40h] [rbp-80h] BYREF
  unsigned int *v62[3]; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v63; // [rsp+68h] [rbp-58h]
  _WORD *v64; // [rsp+70h] [rbp-50h]
  __int64 v65; // [rsp+78h] [rbp-48h]
  unsigned __int64 *v66; // [rsp+80h] [rbp-40h]

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
    v21 = v13 + 1;
    LODWORD(v59) = v21;
    if ( v21 <= a3 )
    {
      while ( 1 )
      {
        v22 = v21 - 1;
        v23 = v6;
        LODWORD(v60[0]) = v22;
        if ( !v5 )
          goto LABEL_31;
        do
        {
          while ( 1 )
          {
            v24 = *(_QWORD *)(v5 + 16);
            v25 = *(_QWORD *)(v5 + 24);
            if ( v22 >= *(_DWORD *)(v5 + 32) )
              break;
            v5 = *(_QWORD *)(v5 + 24);
            if ( !v25 )
              goto LABEL_29;
          }
          v23 = v5;
          v5 = *(_QWORD *)(v5 + 16);
        }
        while ( v24 );
LABEL_29:
        if ( v6 == v23 || v22 > *(_DWORD *)(v23 + 32) )
        {
LABEL_31:
          v62[0] = (unsigned int *)v60;
          v23 = sub_3028E10((_QWORD *)a2, v23, v62);
        }
        v26 = sub_3028300(a2, (_BYTE *)(*(_QWORD *)(v23 + 40) + *(_QWORD *)(v23 + 48)));
        v27 = *(_QWORD *)(a2 + 16);
        v29 = v28;
        if ( !v27 )
          break;
        v30 = (int)v59;
        v31 = v6;
        do
        {
          while ( 1 )
          {
            v32 = *(_QWORD *)(v27 + 16);
            v33 = *(_QWORD *)(v27 + 24);
            if ( *(_DWORD *)(v27 + 32) <= (unsigned int)v59 )
              break;
            v27 = *(_QWORD *)(v27 + 24);
            if ( !v33 )
              goto LABEL_37;
          }
          v31 = v27;
          v27 = *(_QWORD *)(v27 + 16);
        }
        while ( v32 );
LABEL_37:
        if ( v6 == v31 || (unsigned int)v59 > *(_DWORD *)(v31 + 32) )
          goto LABEL_39;
LABEL_40:
        v21 = v30 + 1;
        *(_QWORD *)(v31 + 40) = v26;
        *(_QWORD *)(v31 + 48) = v29;
        *(_BYTE *)(v31 + 56) = 0;
        LODWORD(v59) = v21;
        if ( v21 > a3 )
          goto LABEL_10;
        v5 = *(_QWORD *)(a2 + 16);
      }
      v31 = v6;
LABEL_39:
      v56 = v26;
      v62[0] = (unsigned int *)&v59;
      v34 = sub_3028D50((_QWORD *)a2, v31, v62);
      v30 = (int)v59;
      v26 = v56;
      v31 = v34;
      goto LABEL_40;
    }
  }
LABEL_10:
  v14 = a3;
  if ( !a4 )
  {
    v15 = a3 - 2;
    v16 = a3 - 1;
    if ( a3 == 1 || v16 < v15 )
    {
      v14 = a3;
    }
    else
    {
      while ( 1 )
      {
        v14 = v16 + 1;
        v17 = v6;
        if ( *(_QWORD *)(a2 + 16) )
        {
          v18 = *(_QWORD *)(a2 + 16);
          do
          {
            while ( 1 )
            {
              v19 = *(_QWORD *)(v18 + 16);
              v20 = *(_QWORD *)(v18 + 24);
              if ( *(_DWORD *)(v18 + 32) <= v16 )
                break;
              v18 = *(_QWORD *)(v18 + 24);
              if ( !v20 )
                goto LABEL_18;
            }
            v17 = v18;
            v18 = *(_QWORD *)(v18 + 16);
          }
          while ( v19 );
        }
LABEL_18:
        if ( *(_DWORD *)(v17 + 32) != v16 || *(_BYTE *)(v17 + 56) )
          break;
        if ( v16 - 1 < v15 || v16 == 1 )
        {
          v14 = v16;
          break;
        }
        --v16;
      }
    }
  }
  v60[0] = (unsigned __int64)v61;
  v65 = 0x100000000LL;
  v57 = v14;
  v61[0] = 0;
  v62[0] = (unsigned int *)&unk_49DD210;
  v60[1] = 0;
  v66 = v60;
  v62[1] = 0;
  v62[2] = 0;
  v63 = 0;
  v64 = 0;
  sub_CB5980((__int64)v62, 0, 0, 0);
  v35 = v64;
  v36 = v57;
  if ( (unsigned __int64)v64 >= v63 )
  {
    sub_CB5D20((__int64)v62, 10);
    v36 = v57;
  }
  else
  {
    v64 = (_WORD *)((char *)v64 + 1);
    *v35 = 10;
  }
  v58 = v36;
  while ( v36 <= a3 )
  {
    v37 = *(_QWORD *)(a2 + 16);
    v38 = v6;
    if ( !v37 )
      goto LABEL_53;
    do
    {
      while ( 1 )
      {
        v39 = *(_QWORD *)(v37 + 16);
        v40 = *(_QWORD *)(v37 + 24);
        if ( v36 >= *(_DWORD *)(v37 + 32) )
          break;
        v37 = *(_QWORD *)(v37 + 24);
        if ( !v40 )
          goto LABEL_51;
      }
      v38 = v37;
      v37 = *(_QWORD *)(v37 + 16);
    }
    while ( v39 );
LABEL_51:
    if ( v6 == v38 || v36 > *(_DWORD *)(v38 + 32) )
    {
LABEL_53:
      v59 = &v58;
      v38 = sub_3028D50((_QWORD *)a2, v38, &v59);
    }
    *(_BYTE *)(v38 + 56) = 1;
    if ( v63 - (unsigned __int64)v64 <= 1 )
    {
      v41 = (unsigned int **)sub_CB6200((__int64)v62, (unsigned __int8 *)"//", 2u);
    }
    else
    {
      v41 = v62;
      *v64++ = 12079;
    }
    v42 = sub_CB6200((__int64)v41, *(unsigned __int8 **)(a2 + 64), *(_QWORD *)(a2 + 72));
    v43 = *(_BYTE **)(v42 + 32);
    if ( (unsigned __int64)v43 >= *(_QWORD *)(v42 + 24) )
    {
      v42 = sub_CB5D20(v42, 58);
    }
    else
    {
      *(_QWORD *)(v42 + 32) = v43 + 1;
      *v43 = 58;
    }
    v44 = sub_CB59D0(v42, v58);
    v45 = *(_BYTE **)(v44 + 32);
    if ( (unsigned __int64)v45 >= *(_QWORD *)(v44 + 24) )
    {
      v44 = sub_CB5D20(v44, 32);
      v46 = *(_QWORD *)(a2 + 16);
      if ( v46 )
      {
LABEL_60:
        v47 = v6;
        do
        {
          while ( 1 )
          {
            v48 = *(_QWORD *)(v46 + 16);
            v49 = *(_QWORD *)(v46 + 24);
            if ( *(_DWORD *)(v46 + 32) <= v58 )
              break;
            v46 = *(_QWORD *)(v46 + 24);
            if ( !v49 )
              goto LABEL_64;
          }
          v47 = v46;
          v46 = *(_QWORD *)(v46 + 16);
        }
        while ( v48 );
LABEL_64:
        if ( v6 != v47 && v58 <= *(_DWORD *)(v47 + 32) )
          goto LABEL_67;
        goto LABEL_66;
      }
    }
    else
    {
      *(_QWORD *)(v44 + 32) = v45 + 1;
      *v45 = 32;
      v46 = *(_QWORD *)(a2 + 16);
      if ( v46 )
        goto LABEL_60;
    }
    v47 = v6;
LABEL_66:
    v59 = &v58;
    v47 = sub_3028D50((_QWORD *)a2, v47, &v59);
LABEL_67:
    v50 = *(void **)(v44 + 32);
    v51 = *(_QWORD *)(v47 + 48);
    if ( v51 > *(_QWORD *)(v44 + 24) - (_QWORD)v50 )
    {
      sub_CB6200(v44, *(unsigned __int8 **)(v47 + 40), v51);
    }
    else if ( v51 )
    {
      v54 = *(_QWORD *)(v47 + 48);
      memcpy(v50, *(const void **)(v47 + 40), v51);
      *(_QWORD *)(v44 + 32) += v54;
    }
    v36 = v58 + 1;
    v58 = v36;
  }
  v52 = v66;
  *(_QWORD *)a1 = a1 + 16;
  sub_3020560((__int64 *)a1, (_BYTE *)*v52, *v52 + v52[1]);
  *(_BYTE *)(a1 + 32) = 1;
  v62[0] = (unsigned int *)&unk_49DD210;
  sub_CB5840((__int64)v62);
  if ( (_BYTE *)v60[0] != v61 )
    j_j___libc_free_0(v60[0]);
  return a1;
}
