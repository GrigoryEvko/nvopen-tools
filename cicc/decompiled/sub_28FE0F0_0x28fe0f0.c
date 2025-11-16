// Function: sub_28FE0F0
// Address: 0x28fe0f0
//
__int64 __fastcall sub_28FE0F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // r14
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // rax
  __int64 v10; // r13
  int v11; // ecx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // r13
  _QWORD *v17; // r13
  __int64 v18; // rdi
  _QWORD *v19; // r13
  unsigned __int64 v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // ecx
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rbx
  __int64 v29; // r15
  __int64 v30; // r14
  _QWORD *v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // rbx
  __int64 v34; // rdi
  _QWORD *v35; // rbx
  unsigned __int64 v36; // rdi
  __int64 v38; // rdi
  int v39; // ecx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r14
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp-A0h] [rbp-A0h]
  _QWORD *v46; // [rsp-98h] [rbp-98h]
  _QWORD *v47; // [rsp-88h] [rbp-88h] BYREF
  _QWORD **v48; // [rsp-80h] [rbp-80h]
  __int64 v49; // [rsp-78h] [rbp-78h]
  const char *v50; // [rsp-68h] [rbp-68h] BYREF
  __int16 v51; // [rsp-60h] [rbp-60h]
  char v52; // [rsp-58h] [rbp-58h]
  char v53; // [rsp-48h] [rbp-48h]
  char v54; // [rsp-47h] [rbp-47h]

  v2 = *(_QWORD *)(a1 + 80);
  if ( !v2 )
    BUG();
  v3 = 1;
  v45 = v2 - 24;
  v5 = *(_QWORD *)(v2 + 32);
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    if ( *(_BYTE *)(v5 - 24) != 60 )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    v3 = 0;
  }
  v6 = (_QWORD *)sub_B2BE50(a1);
  v7 = sub_BCB2D0(v6);
  v8 = sub_AD6530(v7, a2);
  v9 = (_QWORD *)sub_B2BE50(a1);
  v54 = 1;
  v10 = sub_BCB2D0(v9);
  v53 = 3;
  v50 = "reg2mem alloca point";
  LODWORD(a2) = dword_3F10A14;
  v46 = sub_BD2C40(72, dword_3F10A14);
  if ( v46 )
  {
    LODWORD(a2) = v8;
    sub_B51BF0((__int64)v46, v8, v10, (__int64)&v50, v5, v3);
  }
  v14 = *(_QWORD *)(a1 + 80);
  v15 = a1 + 72;
  v49 = 0;
  v48 = &v47;
  v47 = &v47;
  if ( a1 + 72 == v14 )
  {
    v16 = 0;
  }
  else
  {
    if ( !v14 )
      BUG();
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 32);
      if ( v16 != v14 + 24 )
        break;
      v14 = *(_QWORD *)(v14 + 8);
      if ( v15 == v14 )
        goto LABEL_14;
      if ( !v14 )
        BUG();
    }
  }
  while ( v14 != v15 )
  {
    if ( !v16 )
      BUG();
    if ( *(_BYTE *)(v16 - 24) != 60 || v45 != *(_QWORD *)(v16 + 16) )
    {
      if ( (v38 = *(_QWORD *)(v16 - 16), v39 = *(unsigned __int8 *)(v38 + 8), (_BYTE)v39 == 12)
        || (unsigned __int8)v39 <= 3u
        || (_BYTE)v39 == 5
        || (v39 & 0xFB) == 0xA
        || (LODWORD(a2) = v39 & 0xFFFFFFFD, (v39 & 0xFD) == 4)
        || ((unsigned __int8)(*(_BYTE *)(v38 + 8) - 15) <= 3u || v39 == 20)
        && (LODWORD(a2) = 0, (unsigned __int8)sub_BCEBA0(v38, 0)) )
      {
        v40 = *(_QWORD *)(v16 - 8);
        a2 = *(_QWORD *)(v16 + 16);
        if ( v40 )
        {
          while ( 1 )
          {
            v41 = *(_QWORD *)(v40 + 24);
            if ( a2 != *(_QWORD *)(v41 + 40) || *(_BYTE *)v41 == 84 )
              break;
            v40 = *(_QWORD *)(v40 + 8);
            if ( !v40 )
              goto LABEL_46;
          }
          v42 = (__int64)v47;
          v43 = (_QWORD *)sub_22077B0(0x18u);
          LODWORD(a2) = v42;
          v43[2] = v16 - 24;
          sub_2208C80(v43, v42);
          ++v49;
        }
      }
    }
LABEL_46:
    v16 = *(_QWORD *)(v16 + 8);
    v11 = 0;
    while ( 1 )
    {
      v44 = v14 - 24;
      if ( !v14 )
        v44 = 0;
      if ( v16 != v44 + 48 )
        break;
      v14 = *(_QWORD *)(v14 + 8);
      if ( v15 == v14 )
        goto LABEL_14;
      if ( !v14 )
        BUG();
      v16 = *(_QWORD *)(v14 + 32);
    }
  }
LABEL_14:
  v17 = v47;
  if ( v47 != &v47 )
  {
    do
    {
      v18 = v17[2];
      v52 = 1;
      v51 = 0;
      LODWORD(a2) = 0;
      v50 = (const char *)(v46 + 3);
      sub_29CC120(v18, 0, 0, v11, v12, v13, (__int64)(v46 + 3), 0, 1);
      v17 = (_QWORD *)*v17;
    }
    while ( v17 != &v47 );
    v19 = v47;
    while ( v19 != &v47 )
    {
      v20 = (unsigned __int64)v19;
      LODWORD(a2) = 24;
      v19 = (_QWORD *)*v19;
      j_j___libc_free_0(v20);
    }
  }
  v21 = *(_QWORD *)(a1 + 80);
  v49 = 0;
  v48 = &v47;
  v47 = &v47;
  if ( v21 != v15 )
  {
    do
    {
      v22 = v21 - 24;
      if ( !v21 )
        v22 = 0;
      v23 = sub_AA5930(v22);
      v28 = v24;
      v29 = v23;
      while ( v28 != v29 )
      {
        v30 = (__int64)v47;
        v31 = (_QWORD *)sub_22077B0(0x18u);
        v31[2] = v29;
        LODWORD(a2) = v30;
        sub_2208C80(v31, v30);
        ++v49;
        if ( !v29 )
          BUG();
        v32 = *(_QWORD *)(v29 + 32);
        if ( !v32 )
          BUG();
        v29 = 0;
        if ( *(_BYTE *)(v32 - 24) == 84 )
          v29 = v32 - 24;
      }
      v21 = *(_QWORD *)(v21 + 8);
    }
    while ( v21 != v15 );
    v33 = v47;
    if ( v47 != &v47 )
    {
      do
      {
        v34 = v33[2];
        v52 = 1;
        v51 = 0;
        v50 = (const char *)(v46 + 3);
        sub_29CBC80(v34, a2, v24, v25, v26, v27, (__int64)(v46 + 3), 0, 1);
        v33 = (_QWORD *)*v33;
      }
      while ( v33 != &v47 );
      v35 = v47;
      while ( v35 != &v47 )
      {
        v36 = (unsigned __int64)v35;
        v35 = (_QWORD *)*v35;
        j_j___libc_free_0(v36);
      }
    }
  }
  return 1;
}
