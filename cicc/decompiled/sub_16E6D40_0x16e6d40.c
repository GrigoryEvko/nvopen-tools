// Function: sub_16E6D40
// Address: 0x16e6d40
//
_QWORD *__fastcall sub_16E6D40(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  __int64 v5; // r12
  int v6; // eax
  char *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rbx
  char *v17; // rsi
  const char *v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // r14
  size_t v24; // rdx
  size_t v25; // r13
  unsigned __int8 *v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _QWORD *v30; // r10
  __int64 v31; // rax
  const char *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // r9d
  _QWORD *v36; // r10
  _QWORD *v37; // r8
  void *v38; // rdi
  __int64 *v39; // rdx
  size_t v40; // rdx
  __int64 v41; // rax
  void *v42; // rax
  __int64 v43; // r14
  __int64 v44; // r9
  _QWORD *v46; // [rsp+10h] [rbp-120h]
  _QWORD *v47; // [rsp+10h] [rbp-120h]
  unsigned int v48; // [rsp+18h] [rbp-118h]
  _QWORD *v49; // [rsp+18h] [rbp-118h]
  _QWORD *v50; // [rsp+18h] [rbp-118h]
  _QWORD *v51; // [rsp+20h] [rbp-110h]
  unsigned int v52; // [rsp+20h] [rbp-110h]
  unsigned int v53; // [rsp+28h] [rbp-108h]
  unsigned __int8 *src; // [rsp+40h] [rbp-F0h]
  void *srca; // [rsp+40h] [rbp-F0h]
  const char *v56; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v57; // [rsp+58h] [rbp-D8h]
  char v58; // [rsp+60h] [rbp-D0h]
  char v59; // [rsp+61h] [rbp-CFh]
  const char *v60; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+78h] [rbp-B8h]
  _BYTE v62[176]; // [rsp+80h] [rbp-B0h] BYREF

  v4 = a1;
  v5 = a3;
  v60 = v62;
  v61 = 0x8000000000LL;
  v6 = *(_DWORD *)(a3 + 32);
  switch ( v6 )
  {
    case 1:
      v7 = (char *)sub_16F8F10(a3, &v60);
      v8 = (unsigned int)v61;
      v10 = v9;
      v11 = (unsigned __int64)v60;
      if ( !(_DWORD)v61 )
        goto LABEL_3;
      goto LABEL_11;
    case 2:
      v8 = *(_QWORD *)(a3 + 80);
      v11 = *(_QWORD *)(a3 + 72);
LABEL_11:
      v56 = (const char *)v11;
      v57 = v8;
      v7 = sub_16E5DA0((__int64)&v56, (char **)(a2 + 112));
      v10 = v14;
LABEL_3:
      v12 = (_QWORD *)sub_22077B0(32);
      if ( v12 )
      {
        v12[1] = v5;
        v12[2] = v7;
        v12[3] = v10;
        *v12 = &unk_49EF930;
      }
      goto LABEL_5;
    case 5:
      v15 = (_QWORD *)sub_22077B0(40);
      v16 = v15;
      if ( v15 )
      {
        v15[1] = v5;
        v15[2] = 0;
        v15[3] = 0;
        *v15 = &unk_49EF980;
        v15[4] = 0;
      }
      *(_BYTE *)(v5 + 76) = 0;
      sub_16FD680(v5);
      if ( !*(_QWORD *)(v5 + 80) )
        v5 = 0;
      while ( 1 )
      {
        if ( !v5 )
          goto LABEL_23;
        sub_16E6D40(&v56, a2, *(_QWORD *)(v5 + 80));
        if ( *(_DWORD *)(a2 + 96) )
          goto LABEL_47;
        v17 = (char *)v16[3];
        if ( v17 == (char *)v16[4] )
          break;
        if ( !v17 )
        {
          v16[3] = 8;
          v18 = v56;
          goto LABEL_25;
        }
        *(_QWORD *)v17 = v56;
        v16[3] += 8LL;
LABEL_22:
        sub_16FD680(v5);
        if ( !*(_QWORD *)(v5 + 80) )
          goto LABEL_23;
      }
      sub_16E6B90(v16 + 2, v17, &v56);
      v18 = v56;
LABEL_25:
      if ( v18 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v18 + 16LL))(v18);
      goto LABEL_22;
  }
  if ( v6 != 4 )
  {
    if ( v6 )
    {
      v59 = 1;
      v56 = "unknown node kind";
      v58 = 3;
      sub_16E4270(a2, a3);
      *a1 = 0;
      goto LABEL_6;
    }
    v12 = (_QWORD *)sub_22077B0(16);
    if ( v12 )
    {
      v12[1] = v5;
      *v12 = &unk_49EF908;
    }
LABEL_5:
    *a1 = v12;
    goto LABEL_6;
  }
  v19 = (_QWORD *)sub_22077B0(256);
  v16 = v19;
  if ( v19 )
  {
    v19[1] = v5;
    v19[2] = 0;
    v19[3] = 0;
    *v19 = &unk_49EF958;
    v19[4] = 0x1000000000LL;
    v19[6] = v19 + 8;
    v19[7] = 0x600000000LL;
  }
  *(_BYTE *)(v5 + 76) = 0;
  sub_16FD380(v5);
  if ( !*(_QWORD *)(v5 + 80) )
    v5 = 0;
  while ( 1 )
  {
    if ( !v5 )
      goto LABEL_44;
    v20 = *(_QWORD *)(v5 + 80);
    v21 = sub_16FD110(v20);
    v22 = v21;
    if ( *(_DWORD *)(v21 + 32) != 1 )
    {
      srca = (void *)v21;
      v4 = a1;
      v43 = sub_16FD200(v20);
      v59 = 1;
      v56 = "Map key must be a scalar";
      v58 = 3;
      sub_16E4270(a2, (__int64)srca);
      v44 = (__int64)srca;
      if ( !v43 )
        goto LABEL_64;
      goto LABEL_23;
    }
    v23 = sub_16FD200(v20);
    if ( !v23 )
      break;
    LODWORD(v61) = 0;
    src = (unsigned __int8 *)sub_16F8F10(v22, &v60);
    v25 = v24;
    if ( (_DWORD)v61 )
    {
      v57 = (unsigned int)v61;
      v56 = v60;
      src = (unsigned __int8 *)sub_16E5DA0((__int64)&v56, (char **)(a2 + 112));
      v25 = v40;
    }
    sub_16E6D40(&v56, a2, v23);
    if ( *(_DWORD *)(a2 + 96) )
    {
      v4 = a1;
LABEL_47:
      if ( v56 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v56 + 16LL))(v56);
      goto LABEL_23;
    }
    v26 = src;
    v29 = (unsigned int)sub_16D19C0((__int64)(v16 + 2), src, v25);
    v30 = (_QWORD *)(v16[2] + 8 * v29);
    v31 = *v30;
    if ( *v30 )
    {
      if ( v31 != -8 )
        goto LABEL_40;
      --*((_DWORD *)v16 + 8);
    }
    v46 = v30;
    v48 = v29;
    v34 = malloc(v25 + 17);
    v35 = v48;
    v36 = v46;
    v37 = (_QWORD *)v34;
    if ( v34 )
      goto LABEL_54;
    if ( v25 != -17 || (v41 = malloc(1u), v35 = v48, v36 = v46, v37 = 0, !v41) )
    {
      v47 = v37;
      v50 = v36;
      v52 = v35;
      sub_16BD1C0("Allocation failed", 1u);
      v35 = v52;
      v36 = v50;
      v37 = v47;
LABEL_54:
      v38 = v37 + 2;
      if ( v25 + 1 <= 1 )
        goto LABEL_55;
      goto LABEL_61;
    }
    v38 = (void *)(v41 + 16);
    v37 = (_QWORD *)v41;
LABEL_61:
    v49 = v37;
    v51 = v36;
    v53 = v35;
    v42 = memcpy(v38, src, v25);
    v37 = v49;
    v36 = v51;
    v35 = v53;
    v38 = v42;
LABEL_55:
    *((_BYTE *)v38 + v25) = 0;
    v26 = (unsigned __int8 *)v35;
    *v37 = v25;
    v37[1] = 0;
    *v36 = v37;
    ++*((_DWORD *)v16 + 7);
    v39 = (__int64 *)(v16[2] + 8LL * (unsigned int)sub_16D1CD0((__int64)(v16 + 2), v35));
    v31 = *v39;
    if ( *v39 != -8 )
      goto LABEL_57;
    do
    {
      do
      {
        v31 = v39[1];
        ++v39;
      }
      while ( v31 == -8 );
LABEL_57:
      ;
    }
    while ( !v31 );
LABEL_40:
    v32 = v56;
    v56 = 0;
    v33 = *(_QWORD *)(v31 + 8);
    *(_QWORD *)(v31 + 8) = v32;
    if ( v33 )
    {
      (*(void (__fastcall **)(__int64, unsigned __int8 *, const char *, __int64, __int64, __int64))(*(_QWORD *)v33 + 16LL))(
        v33,
        v26,
        v32,
        v27,
        v28,
        v29);
      if ( v56 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v56 + 16LL))(v56);
    }
    sub_16FD380(v5);
    if ( !*(_QWORD *)(v5 + 80) )
    {
LABEL_44:
      v4 = a1;
      goto LABEL_23;
    }
  }
  v44 = v22;
  v4 = a1;
LABEL_64:
  v59 = 1;
  v56 = "Map value must not be empty";
  v58 = 3;
  sub_16E4270(a2, v44);
LABEL_23:
  *v4 = v16;
LABEL_6:
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  return v4;
}
