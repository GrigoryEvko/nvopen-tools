// Function: sub_1EAB520
// Address: 0x1eab520
//
__int64 __fastcall sub_1EAB520(__int64 a1, _BYTE **a2)
{
  __int64 (*v2)(void); // rdx
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r13
  int v17; // eax
  _BYTE *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 (*v21)(); // rdx
  void (__fastcall *v22)(__int64, __int64); // rax
  _BYTE *v23; // rdi
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 (*v26)(void); // rdx
  __int64 (*v27)(); // rax
  __int64 v28; // rax
  _BYTE *v29; // rdi
  void (*v30)(); // rax
  __int64 v31; // r14
  __int64 v32; // r14
  _QWORD *v33; // r15
  _QWORD *v34; // r10
  unsigned int v35; // ebx
  unsigned int v36; // r13d
  unsigned __int64 v37; // r9
  __int16 v38; // ax
  unsigned __int64 v39; // rbx
  __int64 i; // rbx
  __int64 v41; // rax
  unsigned int v42; // ebx
  unsigned int v43; // r12d
  __int64 (*v45)(); // rdx
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE **v50; // [rsp+18h] [rbp-968h]
  unsigned __int64 v51; // [rsp+20h] [rbp-960h]
  int v52; // [rsp+2Ch] [rbp-954h]
  unsigned int v53; // [rsp+2Ch] [rbp-954h]
  _BYTE *v54; // [rsp+30h] [rbp-950h] BYREF
  __int64 v55; // [rsp+38h] [rbp-948h]
  _BYTE v56[32]; // [rsp+40h] [rbp-940h] BYREF
  _QWORD v57[263]; // [rsp+60h] [rbp-920h] BYREF
  _QWORD v58[14]; // [rsp+898h] [rbp-E8h] BYREF
  __int64 v59; // [rsp+908h] [rbp-78h]
  __int64 v60; // [rsp+910h] [rbp-70h]
  __int64 v61; // [rsp+918h] [rbp-68h]
  __int64 v62; // [rsp+920h] [rbp-60h]
  __int64 v63; // [rsp+928h] [rbp-58h]
  _QWORD v64[3]; // [rsp+930h] [rbp-50h] BYREF
  unsigned int v65; // [rsp+948h] [rbp-38h]

  v50 = a2;
  v2 = *(__int64 (**)(void))(*(_QWORD *)a2[2] + 40LL);
  v3 = 0;
  if ( v2 != sub_1D00B00 )
    v3 = v2();
  v4 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = v3;
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_85:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FC6A0C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_85;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FC6A0C);
  v8 = *(__int64 **)(a1 + 8);
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_84:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F96DB4 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_84;
  }
  v11 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(
                      *(_QWORD *)(v9 + 8),
                      &unk_4F96DB4)
                  + 160);
  v12 = *(__int64 **)(a1 + 8);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_86:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4FCBA30 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_86;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4FCBA30);
  v16 = a1 + 240;
  sub_1ED7320(a1 + 240);
  v54 = v56;
  v55 = 0x400000000LL;
  v17 = sub_1F45DD0(v15);
  v18 = a2[2];
  LODWORD(v19) = 0;
  v52 = v17;
  v20 = *(_QWORD *)v18;
  v21 = *(__int64 (**)())(*(_QWORD *)v18 + 216LL);
  if ( v21 != sub_1EA9B00 )
  {
    LODWORD(v19) = ((__int64 (__fastcall *)(_BYTE *, _BYTE **))v21)(v18, a2);
    v20 = *(_QWORD *)v18;
  }
  v22 = *(void (__fastcall **)(__int64, __int64))(v20 + 224);
  if ( v22 == sub_1EA9B10 )
  {
    LODWORD(v55) = 0;
  }
  else
  {
    a2 = &v54;
    v22((__int64)v18, (__int64)&v54);
  }
  if ( !(_DWORD)qword_4FC8FB0 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, _BYTE **))(*(_QWORD *)v18 + 176LL))(v18, a2) )
    {
      v45 = *(__int64 (**)())(*(_QWORD *)v18 + 248LL);
      v46 = 2;
      if ( v45 != sub_1EA9B30 )
        v46 = ((__int64 (__fastcall *)(_BYTE *))v45)(v18);
      if ( v46 <= v52 )
        goto LABEL_21;
    }
LABEL_66:
    v43 = 0;
    goto LABEL_59;
  }
  if ( !byte_4FC9040 )
    goto LABEL_66;
LABEL_21:
  if ( (_DWORD)qword_4FC8E90 )
  {
    LODWORD(v19) = 2;
    if ( (unsigned int)sub_2241AC0(&qword_4FC8F20, "all") )
      v19 = (unsigned int)sub_2241AC0(&qword_4FC8F20, "critical") == 0;
  }
  sub_1F03E40(v57, v50, v7, 0);
  v23 = v50[2];
  v24 = 0;
  v57[0] = off_49FD3B8;
  v58[1] = 0;
  v60 = v11;
  v58[0] = &unk_4A00AB0;
  memset(&v58[3], 0, 48);
  v58[9] = v58;
  memset(&v58[10], 0, 24);
  v61 = 0;
  v62 = 0;
  v63 = 0;
  memset(v64, 0, sizeof(v64));
  v65 = 0;
  v25 = *(_QWORD *)v23;
  v26 = *(__int64 (**)(void))(*(_QWORD *)v23 + 128LL);
  if ( v26 != sub_1D0B140 )
  {
    v24 = v26();
    v25 = *(_QWORD *)v50[2];
  }
  v27 = *(__int64 (**)())(v25 + 40);
  if ( v27 == sub_1D00B00 )
    BUG();
  v28 = v27();
  v58[13] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v28 + 768LL))(v28, v24, v57);
  v29 = v50[2];
  v30 = *(void (**)())(*(_QWORD *)v29 + 232LL);
  if ( v30 != nullsub_728 )
    ((void (__fastcall *)(_BYTE *, _QWORD *))v30)(v29, v64);
  if ( (_DWORD)v19 == 2 )
  {
    v48 = sub_22077B0(80);
    v31 = v48;
    if ( v48 )
      sub_20C2AF0(v48, v50, v16, &v54);
  }
  else
  {
    v31 = 0;
    if ( (_DWORD)v19 == 1 )
    {
      v47 = sub_22077B0(216);
      v31 = v47;
      if ( v47 )
        sub_20E5F50(v47, v50, v16);
    }
  }
  v59 = v31;
  v32 = (__int64)v50[41];
  if ( (_BYTE **)v32 != v50 + 40 )
  {
    while ( 1 )
    {
      v33 = (_QWORD *)(v32 + 24);
      sub_1EA9BD0((__int64)v57, v32);
      v34 = *(_QWORD **)(v32 + 32);
      if ( (_QWORD *)(v32 + 24) != v34 )
        break;
      v51 = v32 + 24;
      v42 = 0;
LABEL_53:
      sub_1F03430(v57, v32, v33, v51, v42);
      if ( v61 != v62 )
        v62 = v61;
      v65 = v42;
      sub_1EAAD20((__int64)v57);
      nullsub_753(v57);
      sub_1EA9F10((__int64)v57);
      if ( v59 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v59 + 40LL))(v59);
      sub_1F03420(v57);
      sub_1F04A30(v57, v32);
      v32 = *(_QWORD *)(v32 + 8);
      if ( v50 + 40 == (_BYTE **)v32 )
        goto LABEL_58;
    }
    v35 = 0;
    do
    {
      v34 = (_QWORD *)v34[1];
      ++v35;
    }
    while ( v33 != v34 );
    v51 = (unsigned __int64)v34;
    v36 = v35;
    v33 = v34;
    v53 = v35;
    while ( 1 )
    {
      v37 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v37 )
        BUG();
      v38 = *(_WORD *)(v37 + 46);
      v39 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
      --v36;
      if ( (*(_QWORD *)v37 & 4) != 0 )
      {
        if ( (v38 & 4) != 0 )
          goto LABEL_68;
      }
      else if ( (v38 & 4) != 0 )
      {
        for ( i = *(_QWORD *)v37; ; i = *(_QWORD *)v39 )
        {
          v39 = i & 0xFFFFFFFFFFFFFFF8LL;
          v38 = *(_WORD *)(v39 + 46);
          if ( (v38 & 4) == 0 )
            break;
        }
      }
      if ( (v38 & 8) != 0 )
      {
        LOBYTE(v41) = sub_1E15D00(v39, 0x10u, 1);
        goto LABEL_42;
      }
LABEL_68:
      v41 = (*(_QWORD *)(*(_QWORD *)(v39 + 16) + 8LL) >> 4) & 1LL;
LABEL_42:
      if ( (_BYTE)v41
        || (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, _BYTE **))(**(_QWORD **)(a1 + 232)
                                                                                          + 736LL))(
             *(_QWORD *)(a1 + 232),
             v39,
             v32,
             v50) )
      {
        sub_1F03430(v57, v32, v33, v51, v53 - v36);
        if ( v61 != v62 )
          v62 = v61;
        v65 = v53;
        sub_1EAAD20((__int64)v57);
        nullsub_753(v57);
        sub_1EA9F10((__int64)v57);
        if ( v59 )
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD, _QWORD))(*(_QWORD *)v59 + 32LL))(
            v59,
            v39,
            v36,
            v65);
        v51 = v39;
        v53 = v36;
      }
      if ( **(_WORD **)(v39 + 16) == 16 )
        v36 -= sub_1E16570(v39);
      v33 = (_QWORD *)v39;
      if ( *(_QWORD *)(v32 + 32) == v39 )
      {
        v42 = v53;
        goto LABEL_53;
      }
    }
  }
LABEL_58:
  v43 = 1;
  sub_1EAA280(v57);
LABEL_59:
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return v43;
}
