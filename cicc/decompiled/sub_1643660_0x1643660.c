// Function: sub_1643660
// Address: 0x1643660
//
void __fastcall sub_1643660(__int64 **a1, const void *a2, size_t a3)
{
  const void *v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rsi
  __int64 v9; // r15
  unsigned __int64 v10; // rdi
  unsigned int v11; // r9d
  _QWORD *v12; // r10
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // r12
  __int64 v16; // rax
  _BYTE *v17; // rax
  _BYTE *i; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r15
  size_t v22; // r13
  unsigned int v23; // r8d
  _QWORD *v24; // r10
  __int64 v25; // rax
  const void *v26; // r11
  unsigned int v27; // r8d
  _QWORD *v28; // r10
  _QWORD *v29; // r12
  __int64 **v30; // r12
  __int64 *v31; // rax
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  unsigned int v34; // r9d
  _QWORD *v35; // r10
  _QWORD *v36; // r8
  void *v37; // rdi
  __int64 *v38; // rax
  __int64 v39; // rax
  void *v40; // rax
  _QWORD *v41; // [rsp+10h] [rbp-D0h]
  _QWORD *v42; // [rsp+10h] [rbp-D0h]
  _QWORD *v43; // [rsp+10h] [rbp-D0h]
  unsigned int v44; // [rsp+18h] [rbp-C8h]
  _QWORD *v45; // [rsp+18h] [rbp-C8h]
  unsigned int v46; // [rsp+18h] [rbp-C8h]
  _QWORD *v47; // [rsp+18h] [rbp-C8h]
  _QWORD *v48; // [rsp+18h] [rbp-C8h]
  _QWORD *v49; // [rsp+20h] [rbp-C0h]
  unsigned int v50; // [rsp+20h] [rbp-C0h]
  void *src; // [rsp+28h] [rbp-B8h]
  unsigned int srca; // [rsp+28h] [rbp-B8h]
  unsigned int srcb; // [rsp+28h] [rbp-B8h]
  _QWORD v54[4]; // [rsp+30h] [rbp-B0h] BYREF
  int v55; // [rsp+50h] [rbp-90h]
  _BYTE **v56; // [rsp+58h] [rbp-88h]
  _BYTE *v57; // [rsp+60h] [rbp-80h] BYREF
  __int64 v58; // [rsp+68h] [rbp-78h]
  _BYTE dest[112]; // [rsp+70h] [rbp-70h] BYREF

  v6 = (const void *)sub_1643640((__int64)a1);
  if ( v7 == a3 && (!a3 || !memcmp(a2, v6, a3)) )
    return;
  v8 = a1[3];
  v9 = **a1;
  if ( v8 )
  {
    sub_16D1CB0(v9 + 2472, v8);
    if ( !a3 )
    {
      v10 = (unsigned __int64)a1[3];
      if ( v10 )
      {
        _libc_free(v10);
        a1[3] = 0;
      }
      return;
    }
    v9 = **a1;
  }
  else if ( !a3 )
  {
    return;
  }
  v11 = sub_16D19C0(v9 + 2472, a2, a3);
  v12 = (_QWORD *)(*(_QWORD *)(v9 + 2472) + 8LL * v11);
  if ( !*v12 )
  {
LABEL_44:
    v42 = v12;
    v46 = v11;
    v33 = malloc(a3 + 17);
    v34 = v46;
    v35 = v42;
    v36 = (_QWORD *)v33;
    if ( !v33 )
    {
      if ( a3 == -17 )
      {
        v39 = malloc(1u);
        v34 = v46;
        v35 = v42;
        v36 = 0;
        if ( v39 )
        {
          v37 = (void *)(v39 + 16);
          v36 = (_QWORD *)v39;
LABEL_53:
          v47 = v36;
          v49 = v35;
          srcb = v34;
          v40 = memcpy(v37, a2, a3);
          v36 = v47;
          v35 = v49;
          v34 = srcb;
          v37 = v40;
LABEL_46:
          *((_BYTE *)v37 + a3) = 0;
          *v36 = a3;
          v36[1] = a1;
          *v35 = v36;
          ++*(_DWORD *)(v9 + 2484);
          v30 = (__int64 **)(*(_QWORD *)(v9 + 2472) + 8LL * (unsigned int)sub_16D1CD0(v9 + 2472, v34));
          v38 = *v30;
          if ( *v30 != (__int64 *)-8LL )
            goto LABEL_48;
          while ( 1 )
          {
            do
            {
              v38 = v30[1];
              ++v30;
            }
            while ( v38 == (__int64 *)-8LL );
LABEL_48:
            if ( v38 )
              goto LABEL_37;
          }
        }
      }
      v43 = v36;
      v48 = v35;
      v50 = v34;
      sub_16BD1C0("Allocation failed");
      v34 = v50;
      v35 = v48;
      v36 = v43;
    }
    v37 = v36 + 2;
    if ( a3 + 1 <= 1 )
      goto LABEL_46;
    goto LABEL_53;
  }
  if ( *v12 == -8 )
  {
    --*(_DWORD *)(v9 + 2488);
    goto LABEL_44;
  }
  v13 = dest;
  v57 = dest;
  v58 = 0x4000000000LL;
  if ( a3 > 0x40 )
  {
    sub_16CD150(&v57, dest, a3, 1);
    v13 = &v57[(unsigned int)v58];
  }
  memcpy(v13, a2, a3);
  LODWORD(v58) = a3 + v58;
  v14 = (unsigned int)v58;
  if ( HIDWORD(v58) <= (unsigned int)v58 )
  {
    sub_16CD150(&v57, dest, 0, 1);
    v14 = (unsigned int)v58;
  }
  v15 = (unsigned int)(a3 + 1);
  v57[v14] = 46;
  LODWORD(v58) = v58 + 1;
  v55 = 1;
  v54[0] = &unk_49EFC48;
  v56 = &v57;
  memset(&v54[1], 0, 24);
  sub_16E7A40(v54, 0, 0, 0);
  do
  {
    v16 = (unsigned int)v58;
    if ( (unsigned int)v58 <= v15 )
    {
      if ( (unsigned int)v58 >= v15 )
        goto LABEL_24;
      if ( HIDWORD(v58) < (unsigned int)v15 )
      {
        sub_16CD150(&v57, dest, v15, 1);
        v16 = (unsigned int)v58;
      }
      v17 = &v57[v16];
      for ( i = &v57[v15]; i != v17; ++v17 )
      {
        if ( v17 )
          *v17 = 0;
      }
    }
    LODWORD(v58) = v15;
LABEL_24:
    v19 = **a1;
    v20 = *(unsigned int *)(v19 + 2504);
    *(_DWORD *)(v19 + 2504) = v20 + 1;
    sub_16E7A90(v54, v20);
    v21 = **a1;
    v22 = *((unsigned int *)v56 + 2);
    src = *v56;
    v23 = sub_16D19C0(v21 + 2472, *v56, v22);
    v24 = (_QWORD *)(*(_QWORD *)(v21 + 2472) + 8LL * v23);
    if ( !*v24 )
      goto LABEL_27;
  }
  while ( *v24 != -8 );
  --*(_DWORD *)(v21 + 2488);
LABEL_27:
  v41 = v24;
  v44 = v23;
  v25 = malloc(v22 + 17);
  v26 = src;
  v27 = v44;
  v28 = v41;
  v29 = (_QWORD *)v25;
  if ( !v25 )
  {
    sub_16BD1C0("Allocation failed");
    v28 = v41;
    v27 = v44;
    v26 = src;
  }
  if ( v22 )
  {
    v45 = v28;
    srca = v27;
    memcpy(v29 + 2, v26, v22);
    v28 = v45;
    v27 = srca;
  }
  *((_BYTE *)v29 + v22 + 16) = 0;
  *v29 = v22;
  v29[1] = a1;
  *v28 = v29;
  ++*(_DWORD *)(v21 + 2484);
  v30 = (__int64 **)(*(_QWORD *)(v21 + 2472) + 8LL * (unsigned int)sub_16D1CD0(v21 + 2472, v27));
  if ( !*v30 || *v30 == (__int64 *)-8LL )
  {
    do
    {
      do
      {
        v31 = v30[1];
        ++v30;
      }
      while ( v31 == (__int64 *)-8LL );
    }
    while ( !v31 );
  }
  v54[0] = &unk_49EFD28;
  sub_16E7960(v54);
  if ( v57 != dest )
    _libc_free((unsigned __int64)v57);
LABEL_37:
  v32 = (unsigned __int64)a1[3];
  if ( v32 )
    _libc_free(v32);
  a1[3] = *v30;
}
