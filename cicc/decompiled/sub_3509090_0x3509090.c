// Function: sub_3509090
// Address: 0x3509090
//
void __fastcall sub_3509090(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  unsigned int v9; // r12d
  void *v10; // rax
  _QWORD *v11; // r14
  unsigned __int64 v12; // r15
  __int64 i; // r15
  unsigned __int64 j; // r12
  __int64 v15; // r8
  __int64 v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rcx
  unsigned int v21; // r9d
  char v22; // al
  int v23; // r9d
  char v24; // r8
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  __int64 v30; // r13
  __int64 v31; // rcx
  __int64 v32; // r12
  char v33; // al
  unsigned int v34; // edx
  char v35; // al
  __int64 v36; // rcx
  __int64 v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  __int64 v40; // rax
  __int64 k; // r15
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  int v46; // [rsp+2Ch] [rbp-84h]
  __int64 v47; // [rsp+30h] [rbp-80h]
  char v48; // [rsp+30h] [rbp-80h]
  __int64 v49; // [rsp+30h] [rbp-80h]
  __int64 v50; // [rsp+38h] [rbp-78h]
  __int64 v51; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v52; // [rsp+48h] [rbp-68h]
  __int64 v53; // [rsp+50h] [rbp-60h]
  __int64 v54; // [rsp+58h] [rbp-58h]
  _BYTE v55[16]; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v56; // [rsp+70h] [rbp-40h]
  unsigned int v57; // [rsp+78h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 32);
  v50 = *(_QWORD *)(v2 + 32);
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)v50 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)v50 + 16LL));
  v8 = *(_QWORD *)(v2 + 48);
  v53 = 0;
  v54 = 8;
  v56 = 0;
  v57 = 0;
  v9 = *(_DWORD *)(v3 + 16);
  v52 = v55;
  v51 = v3;
  if ( v9 )
  {
    v10 = _libc_calloc(v9, 1u);
    if ( !v10 )
      sub_C64F00("Allocation failed", 1u);
    v56 = (unsigned __int64)v10;
    v57 = v9;
  }
  v11 = &v51;
  sub_35085F0(&v51, a1, v4, v5, v6, v7);
  v45 = a1 + 48;
  if ( (*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
LABEL_81:
    BUG();
  v12 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)v12 & 4) == 0 && (*(_BYTE *)(v12 + 44) & 4) != 0 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL); ; i = *(_QWORD *)v12 )
    {
      v12 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
        break;
    }
  }
LABEL_10:
  if ( v45 == v12 )
    goto LABEL_63;
  do
  {
    for ( j = v12; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
      ;
    while ( 1 )
    {
      v15 = *(_QWORD *)(j + 32);
      v16 = v15 + 40LL * (*(_DWORD *)(j + 40) & 0xFFFFFF);
      if ( v15 != v16 )
        break;
      j = *(_QWORD *)(j + 8);
      if ( *(_QWORD *)(v12 + 24) + 48LL == j )
        break;
      if ( (*(_BYTE *)(j + 44) & 4) == 0 )
      {
        j = *(_QWORD *)(v12 + 24) + 48LL;
        break;
      }
    }
    v17 = v11;
    v18 = v15;
    v19 = *(_QWORD *)(v12 + 24) + 48LL;
    v20 = (__int64)v17;
    while ( v16 != v18 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v18 && (*(_BYTE *)(v18 + 3) & 0x10) != 0 && (*(_BYTE *)(v18 + 4) & 8) == 0 )
        {
          v21 = *(_DWORD *)(v18 + 8);
          if ( v21 )
          {
            v47 = v20;
            v46 = *(_DWORD *)(v18 + 8);
            v22 = sub_35080D0(v20, v50, v21);
            v20 = v47;
            v23 = v46;
            v24 = v22;
            v25 = *(_DWORD *)(v12 + 44);
            if ( (v25 & 4) != 0 || (v25 & 8) == 0 )
            {
              v26 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 5) & 1LL;
            }
            else
            {
              v44 = v47;
              v48 = v24;
              LOBYTE(v26) = sub_2E88A90(v12, 32, 1);
              v24 = v48;
              v23 = v46;
              v20 = v44;
            }
            if ( (_BYTE)v26 )
            {
              if ( *(_BYTE *)(v8 + 120) )
              {
                v42 = *(_QWORD *)(v8 + 96);
                v43 = *(_QWORD *)(v8 + 104);
                if ( v43 != v42 )
                {
                  while ( v23 != *(_DWORD *)v42 )
                  {
                    v42 += 12;
                    if ( v43 == v42 )
                      goto LABEL_27;
                  }
                  v24 = *(_BYTE *)(v42 + 8) ^ 1;
                }
              }
            }
LABEL_27:
            *(_BYTE *)(v18 + 3) = *(_BYTE *)(v18 + 3) & 0xBF | ((v24 & 1) << 6);
          }
        }
        v27 = v18 + 40;
        v28 = v16;
        if ( v27 == v16 )
          break;
        v16 = v27;
LABEL_69:
        v18 = v16;
        v16 = v28;
      }
      while ( 1 )
      {
        j = *(_QWORD *)(j + 8);
        if ( v19 == j )
        {
          v18 = v16;
          v16 = v28;
          goto LABEL_34;
        }
        if ( (*(_BYTE *)(j + 44) & 4) == 0 )
          break;
        v16 = *(_QWORD *)(j + 32);
        v28 = v16 + 40LL * (*(_DWORD *)(j + 40) & 0xFFFFFF);
        if ( v16 != v28 )
          goto LABEL_69;
      }
      v18 = v16;
      j = v19;
      v16 = v28;
LABEL_34:
      ;
    }
    v29 = v12;
    v11 = (_QWORD *)v20;
    sub_3508B80(v20, v12);
    if ( (*(_BYTE *)(v12 + 44) & 4) != 0 )
    {
      do
        v29 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v29 + 44) & 4) != 0 );
    }
    v30 = *(_QWORD *)(v12 + 24) + 48LL;
    while ( 1 )
    {
      v31 = *(_QWORD *)(v29 + 32);
      v32 = v31 + 40LL * (*(_DWORD *)(v29 + 40) & 0xFFFFFF);
      if ( v31 != v32 )
        break;
      v29 = *(_QWORD *)(v29 + 8);
      if ( v30 == v29 )
        goto LABEL_55;
      if ( (*(_BYTE *)(v29 + 44) & 4) == 0 )
      {
        v29 = *(_QWORD *)(v12 + 24) + 48LL;
        goto LABEL_55;
      }
    }
    while ( 1 )
    {
LABEL_41:
      if ( !*(_BYTE *)v31 )
      {
        v33 = *(_BYTE *)(v31 + 4);
        if ( (v33 & 1) == 0
          && (v33 & 2) == 0
          && ((*(_BYTE *)(v31 + 3) & 0x10) == 0 || (*(_DWORD *)v31 & 0xFFF00) != 0)
          && (v33 & 8) == 0 )
        {
          v34 = *(_DWORD *)(v31 + 8);
          if ( v34 )
          {
            v49 = v31;
            v35 = sub_35080D0((__int64)v11, v50, v34);
            v31 = v49;
            *(_BYTE *)(v49 + 3) = ((v35 & 1) << 6) | *(_BYTE *)(v49 + 3) & 0xBF;
          }
        }
      }
      v36 = v31 + 40;
      v37 = v32;
      if ( v36 == v32 )
        break;
      v32 = v36;
LABEL_71:
      v31 = v32;
      v32 = v37;
    }
    while ( 1 )
    {
      v29 = *(_QWORD *)(v29 + 8);
      if ( v30 == v29 )
        break;
      if ( (*(_BYTE *)(v29 + 44) & 4) == 0 )
      {
        v29 = v30;
        break;
      }
      v32 = *(_QWORD *)(v29 + 32);
      v37 = v32 + 40LL * (*(_DWORD *)(v29 + 40) & 0xFFFFFF);
      if ( v32 != v37 )
        goto LABEL_71;
    }
    v31 = v32;
    v32 = v37;
LABEL_55:
    if ( v31 != v32 )
      goto LABEL_41;
    sub_3508890(v11, v12);
    v38 = (_QWORD *)(*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL);
    v39 = v38;
    if ( !v38 )
      goto LABEL_81;
    v12 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
    v40 = *v38;
    if ( (v40 & 4) != 0 || (*((_BYTE *)v39 + 44) & 4) == 0 )
      goto LABEL_10;
    for ( k = v40; ; k = *(_QWORD *)v12 )
    {
      v12 = k & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
        break;
    }
  }
  while ( v45 != v12 );
LABEL_63:
  if ( v56 )
    _libc_free(v56);
  if ( v52 != v55 )
    _libc_free((unsigned __int64)v52);
}
