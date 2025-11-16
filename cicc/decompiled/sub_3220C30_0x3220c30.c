// Function: sub_3220C30
// Address: 0x3220c30
//
void __fastcall sub_3220C30(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int8 v4; // al
  _BYTE *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int16 v11; // ax
  _BOOL8 v12; // r14
  _BOOL8 v13; // r14
  const char *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  __int64 v18; // r8
  void (*v19)(); // rax
  _QWORD *v20; // r12
  __int64 v21; // rdi
  void (*v22)(); // rax
  void (__fastcall *v23)(_QWORD *, _QWORD, _QWORD, _QWORD); // r13
  __int64 v24; // rax
  bool v25; // zf
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  void (*v28)(); // r14
  const char *v29; // rax
  __int64 v30; // r8
  __int64 v31; // rdx
  _QWORD *v32; // rdi
  __int64 v33; // r8
  void (*v34)(); // rax
  __int64 v35; // rdi
  void (*v36)(); // rax
  _BOOL8 v37; // r14
  const char *v38; // rax
  __int64 v39; // r8
  __int64 v40; // rdx
  _QWORD *v41; // rdi
  __int64 v42; // r8
  void (*v43)(); // rax
  __int64 v44; // r12
  __int64 v45; // rdi
  void (*v46)(); // rax
  __int64 v47; // rax
  char v48; // dl
  __int64 *v49; // rax
  __int64 v50; // [rsp+0h] [rbp-C0h]
  __int64 v51; // [rsp+0h] [rbp-C0h]
  void (*v52)(); // [rsp+8h] [rbp-B8h]
  __int64 v53; // [rsp+8h] [rbp-B8h]
  void (*v54)(); // [rsp+8h] [rbp-B8h]
  _QWORD *v55; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v56; // [rsp+18h] [rbp-A8h]
  _BYTE v57[16]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v58[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v59; // [rsp+50h] [rbp-70h]
  const char *v60; // [rsp+60h] [rbp-60h] BYREF
  __int64 v61; // [rsp+68h] [rbp-58h]
  __int64 v62; // [rsp+70h] [rbp-50h]
  __int64 v63; // [rsp+78h] [rbp-48h]
  __int16 v64; // [rsp+80h] [rbp-40h]

  v2 = a2 - 16;
  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = **(_BYTE ***)(a2 - 32);
    if ( !v5 )
    {
      v7 = 0;
      goto LABEL_19;
    }
LABEL_3:
    v5 = (_BYTE *)sub_B91420((__int64)v5);
    v4 = *(_BYTE *)(a2 - 16);
    v7 = v6;
    if ( (v4 & 2) == 0 )
      goto LABEL_4;
LABEL_19:
    v8 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( !v8 )
      goto LABEL_20;
    goto LABEL_5;
  }
  v5 = *(_BYTE **)(v2 - 8LL * ((v4 >> 2) & 0xF));
  if ( v5 )
    goto LABEL_3;
  v7 = 0;
LABEL_4:
  v8 = *(_QWORD *)(v2 - 8LL * ((v4 >> 2) & 0xF) + 8);
  if ( !v8 )
    goto LABEL_20;
LABEL_5:
  v9 = sub_B91420(v8);
  if ( v10 )
  {
    v62 = v9;
    v58[2] = " ";
    v59 = 773;
    v58[0] = v5;
    v58[1] = v7;
    v60 = (const char *)v58;
    v63 = v10;
    v64 = 1282;
    sub_CA0F50((__int64 *)&v55, (void **)&v60);
    goto LABEL_7;
  }
LABEL_20:
  if ( !v5 )
  {
    v57[0] = 0;
    v55 = v57;
    v56 = 0;
LABEL_7:
    if ( *(_BYTE *)(a1 + 3692) )
      goto LABEL_8;
    goto LABEL_22;
  }
  v55 = v57;
  sub_3219430((__int64 *)&v55, v5, (__int64)&v5[v7]);
  if ( *(_BYTE *)(a1 + 3692) )
  {
LABEL_8:
    v11 = sub_3220AA0(a1);
    v12 = *(_WORD *)(a2 + 2) != 1;
    if ( v11 <= 4u )
    {
      v37 = v12 + 5;
      v51 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
      v54 = *(void (**)())(*(_QWORD *)v51 + 120LL);
      v38 = sub_E0C7F0((unsigned int)(*(_WORD *)(a2 + 2) != 1) + 5);
      v64 = 261;
      v39 = v51;
      v60 = v38;
      v61 = v40;
      if ( v54 != nullsub_98 )
        ((void (__fastcall *)(__int64, const char **, __int64))v54)(v51, &v60, 1);
      (*(void (__fastcall **)(_QWORD, _BOOL8, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 424LL))(
        *(_QWORD *)(a1 + 8),
        v37,
        0,
        0,
        v39);
      v41 = *(_QWORD **)(a1 + 8);
      v42 = v41[28];
      v43 = *(void (**)())(*(_QWORD *)v42 + 120LL);
      v60 = "Line Number";
      v64 = 259;
      if ( v43 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, const char **, __int64))v43)(v42, &v60, 1);
        v41 = *(_QWORD **)(a1 + 8);
      }
      (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v41 + 424LL))(v41, *(unsigned int *)(a2 + 4), 0, 0);
      v44 = *(_QWORD *)(a1 + 8);
      v45 = *(_QWORD *)(v44 + 224);
      v46 = *(void (**)())(*(_QWORD *)v45 + 120LL);
      v60 = "Macro String";
      v64 = 259;
      if ( v46 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, const char **, __int64))v46)(v45, &v60, 1);
        v44 = *(_QWORD *)(a1 + 8);
      }
      v47 = sub_3247180(a1 + 3256, v44, v55, v56);
      v48 = v47;
      v49 = (__int64 *)(v47 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v48 & 4) == 0 )
        ++v49;
      sub_31F0D70(v44, *v49, 0);
    }
    else
    {
      v13 = v12 + 11;
      v50 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
      v52 = *(void (**)())(*(_QWORD *)v50 + 120LL);
      v14 = sub_E0C700((unsigned int)(*(_WORD *)(a2 + 2) != 1) + 11);
      v64 = 261;
      v15 = v50;
      v60 = v14;
      v61 = v16;
      if ( v52 != nullsub_98 )
        ((void (__fastcall *)(__int64, const char **, __int64))v52)(v50, &v60, 1);
      (*(void (__fastcall **)(_QWORD, _BOOL8, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 424LL))(
        *(_QWORD *)(a1 + 8),
        v13,
        0,
        0,
        v15);
      v17 = *(_QWORD **)(a1 + 8);
      v18 = v17[28];
      v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
      v60 = "Line Number";
      v64 = 259;
      if ( v19 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, const char **, __int64))v19)(v18, &v60, 1);
        v17 = *(_QWORD **)(a1 + 8);
      }
      (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v17 + 424LL))(v17, *(unsigned int *)(a2 + 4), 0, 0);
      v20 = *(_QWORD **)(a1 + 8);
      v21 = v20[28];
      v22 = *(void (**)())(*(_QWORD *)v21 + 120LL);
      v60 = "Macro String";
      v64 = 259;
      if ( v22 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, const char **, __int64))v22)(v21, &v60, 1);
        v20 = *(_QWORD **)(a1 + 8);
      }
      v23 = *(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v20 + 424LL);
      v24 = sub_3247190(a1 + 3256, v20, v55, v56);
      v25 = (v24 & 4) == 0;
      v26 = (v24 & 0xFFFFFFFFFFFFFFF8LL) + 8;
      v27 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v25 )
        v27 = v26;
      v23(v20, *(unsigned int *)(v27 + 16), 0, 0);
    }
    goto LABEL_29;
  }
LABEL_22:
  v53 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v28 = *(void (**)())(*(_QWORD *)v53 + 120LL);
  v29 = sub_E0C510(*(unsigned __int16 *)(a2 + 2));
  v30 = v53;
  v64 = 261;
  v60 = v29;
  v61 = v31;
  if ( v28 != nullsub_98 )
    ((void (__fastcall *)(__int64, const char **, __int64))v28)(v53, &v60, 1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 424LL))(
    *(_QWORD *)(a1 + 8),
    *(unsigned __int16 *)(a2 + 2),
    0,
    0,
    v30);
  v32 = *(_QWORD **)(a1 + 8);
  v33 = v32[28];
  v34 = *(void (**)())(*(_QWORD *)v33 + 120LL);
  v60 = "Line Number";
  v64 = 259;
  if ( v34 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v34)(v33, &v60, 1);
    v32 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v32 + 424LL))(v32, *(unsigned int *)(a2 + 4), 0, 0);
  v35 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v36 = *(void (**)())(*(_QWORD *)v35 + 120LL);
  v60 = "Macro String";
  v64 = 259;
  if ( v36 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v36)(v35, &v60, 1);
    v35 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  }
  (*(void (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v35 + 512LL))(v35, v55, v56);
  sub_31DC9D0(*(_QWORD *)(a1 + 8), 0);
LABEL_29:
  if ( v55 != (_QWORD *)v57 )
    j_j___libc_free_0((unsigned __int64)v55);
}
