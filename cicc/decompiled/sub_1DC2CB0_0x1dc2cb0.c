// Function: sub_1DC2CB0
// Address: 0x1dc2cb0
//
void __fastcall sub_1DC2CB0(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  unsigned int v8; // r14d
  void *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r9d
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  unsigned __int64 i; // r14
  __int64 v17; // r10
  __int64 v18; // r11
  __int64 v19; // r8
  char v20; // al
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // r14
  char v29; // al
  char v30; // al
  __int64 *v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  _QWORD *v35; // [rsp+10h] [rbp-A0h]
  __int64 v36; // [rsp+18h] [rbp-98h]
  __int64 v37; // [rsp+20h] [rbp-90h]
  __int64 v38; // [rsp+20h] [rbp-90h]
  __int64 v39; // [rsp+28h] [rbp-88h]
  __int64 v40; // [rsp+28h] [rbp-88h]
  __int64 v41; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v42; // [rsp+38h] [rbp-78h]
  __int64 v43; // [rsp+40h] [rbp-70h]
  _BYTE v44[32]; // [rsp+48h] [rbp-68h] BYREF
  unsigned __int64 v45; // [rsp+68h] [rbp-48h]
  unsigned int v46; // [rsp+70h] [rbp-40h]

  v1 = *(_QWORD *)(a1[7] + 40LL);
  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)v1 + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 )
  {
    v45 = 0;
    v42 = v44;
    v46 = 0;
    v43 = 0x800000000LL;
    v41 = 0;
    BUG();
  }
  v3 = v2();
  v46 = 0;
  v42 = v44;
  v43 = 0x800000000LL;
  v45 = 0;
  v8 = *(_DWORD *)(v3 + 16);
  v41 = v3;
  if ( v8 )
  {
    v9 = _libc_calloc(v8, 1u);
    if ( !v9 )
      sub_16BD1C0("Allocation failed", 1u);
    v45 = (unsigned __int64)v9;
    v46 = v8;
  }
  sub_1DC29C0(&v41, a1, v4, v5, v6, v7);
  v35 = a1 + 3;
  if ( (a1[3] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    BUG();
  v13 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  v14 = *(_QWORD *)v13;
  if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v34 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = v34;
      if ( (*(_BYTE *)(v34 + 46) & 4) == 0 )
        break;
      v14 = *(_QWORD *)v34;
    }
  }
LABEL_8:
  if ( v35 == (_QWORD *)v13 )
    goto LABEL_54;
  do
  {
    v15 = v13;
    for ( i = v13; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v17 = *(_QWORD *)(v13 + 24) + 24LL;
    do
    {
      v18 = *(_QWORD *)(i + 32);
      v19 = v18 + 40LL * *(unsigned int *)(i + 40);
      if ( v18 != v19 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( v17 == i )
        break;
    }
    while ( (*(_BYTE *)(i + 46) & 4) != 0 );
    if ( v19 != v18 )
    {
      do
      {
        while ( 1 )
        {
          if ( !*(_BYTE *)v18 && (*(_BYTE *)(v18 + 3) & 0x10) != 0 && (*(_BYTE *)(v18 + 4) & 8) == 0 )
          {
            v10 = *(unsigned int *)(v18 + 8);
            if ( (_DWORD)v10 )
            {
              v36 = v19;
              v37 = v18;
              v39 = v17;
              v20 = sub_1DC24A0(&v41, v1, v10);
              v18 = v37;
              v19 = v36;
              v17 = v39;
              v10 = (unsigned __int8)(v20 & 1) << 6;
              *(_BYTE *)(v37 + 3) = ((v20 & 1) << 6) | *(_BYTE *)(v37 + 3) & 0xBF;
            }
          }
          v21 = v18 + 40;
          v22 = v19;
          if ( v21 == v19 )
            break;
          v19 = v21;
LABEL_57:
          v18 = v19;
          v19 = v22;
        }
        while ( 1 )
        {
          i = *(_QWORD *)(i + 8);
          if ( v17 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
            break;
          v19 = *(_QWORD *)(i + 32);
          v22 = v19 + 40LL * *(unsigned int *)(i + 40);
          if ( v19 != v22 )
            goto LABEL_57;
        }
        v18 = v19;
        v19 = v22;
      }
      while ( v22 != v18 );
    }
    sub_1DC2010((__int64)&v41, v13, v10, v11, v19, v12);
    if ( (*(_BYTE *)(v13 + 46) & 4) != 0 )
    {
      do
        v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v15 + 46) & 4) != 0 );
    }
    v26 = *(_QWORD *)(v13 + 24) + 24LL;
    while ( 1 )
    {
      v27 = *(_QWORD *)(v15 + 32);
      v28 = v27 + 40LL * *(unsigned int *)(v15 + 40);
      if ( v27 != v28 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v26 == v15 || (*(_BYTE *)(v15 + 46) & 4) == 0 )
        goto LABEL_46;
    }
    do
    {
LABEL_33:
      if ( !*(_BYTE *)v27 )
      {
        v29 = *(_BYTE *)(v27 + 4);
        if ( (v29 & 1) == 0
          && (v29 & 2) == 0
          && ((*(_BYTE *)(v27 + 3) & 0x10) == 0 || (*(_DWORD *)v27 & 0xFFF00) != 0)
          && (v29 & 8) == 0 )
        {
          v23 = *(unsigned int *)(v27 + 8);
          if ( (_DWORD)v23 )
          {
            v38 = v26;
            v40 = v27;
            v30 = sub_1DC24A0(&v41, v1, v23);
            v27 = v40;
            v26 = v38;
            v23 = (unsigned __int8)(v30 & 1) << 6;
            *(_BYTE *)(v40 + 3) = ((v30 & 1) << 6) | *(_BYTE *)(v40 + 3) & 0xBF;
          }
        }
      }
      v27 += 40;
    }
    while ( v28 != v27 );
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( v26 == v15 || (*(_BYTE *)(v15 + 46) & 4) == 0 )
        break;
      v27 = *(_QWORD *)(v15 + 32);
      v28 = v27 + 40LL * *(unsigned int *)(v15 + 40);
      if ( v27 != v28 )
        goto LABEL_33;
    }
LABEL_46:
    if ( v27 != v28 )
      goto LABEL_33;
    sub_1DC2130(&v41, v13, v23, v27, v24, v25);
    v31 = (__int64 *)(*(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL);
    v10 = (__int64)v31;
    if ( !v31 )
      BUG();
    v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
    v32 = *v31;
    if ( (v32 & 4) != 0 || (*(_BYTE *)(v10 + 46) & 4) == 0 )
      goto LABEL_8;
    while ( 1 )
    {
      v33 = v32 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = v33;
      if ( (*(_BYTE *)(v33 + 46) & 4) == 0 )
        break;
      v32 = *(_QWORD *)v33;
    }
  }
  while ( v35 != (_QWORD *)v33 );
LABEL_54:
  _libc_free(v45);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
}
