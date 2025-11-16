// Function: sub_3726A90
// Address: 0x3726a90
//
__int64 __fastcall sub_3726A90(unsigned __int16 *a1)
{
  __int64 v2; // r15
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r9
  void (*v9)(); // r8
  char v10; // dl
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // r9
  void (*v18)(); // r8
  char v19; // dl
  __int64 v20; // rsi
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // r15
  int v25; // r13d
  int v26; // esi
  __int64 v27; // rdi
  __int64 v28; // r10
  void (*v29)(); // r9
  __int64 *v30; // r15
  __int64 v31; // rdi
  __int64 v32; // r8
  __int64 v33; // r13
  void (*v34)(); // rax
  void (*v35)(); // rbx
  const char *v36; // rax
  __int64 v37; // r8
  __int64 v38; // rdx
  unsigned int *v39; // rbx
  unsigned int *v40; // r13
  __int64 v41; // r15
  unsigned int v42; // edi
  void (__fastcall *v43)(__int64, _QWORD, const char *, _QWORD); // r14
  const char *v44; // rax
  __int64 v45; // r14
  void (__fastcall *v46)(__int64, _QWORD, const char *, _QWORD); // r15
  const char *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 *v53; // [rsp+10h] [rbp-90h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  _QWORD *v55; // [rsp+20h] [rbp-80h]
  _QWORD *v56; // [rsp+20h] [rbp-80h]
  __int64 *v57; // [rsp+20h] [rbp-80h]
  __int64 v58; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v59[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v60; // [rsp+50h] [rbp-50h]
  __int16 v61; // [rsp+60h] [rbp-40h]

  sub_3721BD0(a1 + 10, (__int64 *)a1);
  v2 = *((_QWORD *)a1 + 29);
  v3 = 16LL * *((_QWORD *)a1 + 30);
  v4 = v2 + v3;
  if ( v2 != v2 + v3 )
  {
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)a1;
        v8 = *(_QWORD *)(*(_QWORD *)a1 + 224LL);
        v9 = *(void (**)())(*(_QWORD *)v8 + 120LL);
        v58 = v5;
        v59[0] = "Compilation unit ";
        v60 = &v58;
        v61 = 2819;
        if ( v9 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v9)(v8, v59, 1);
          v7 = *(_QWORD *)a1;
        }
        v10 = *(_BYTE *)(v2 + 8);
        if ( v10 )
          break;
        v11 = *(_QWORD *)v2;
        v2 += 16;
        ++v5;
        sub_31F0D70(v7, v11, 0);
        if ( v4 == v2 )
          goto LABEL_9;
      }
      if ( v10 != 1 )
LABEL_38:
        abort();
      v6 = *(_QWORD *)v2;
      v2 += 16;
      ++v5;
      sub_31F0F00(v7, v6);
    }
    while ( v4 != v2 );
  }
LABEL_9:
  v12 = *((_QWORD *)a1 + 31);
  v13 = 16LL * *((_QWORD *)a1 + 32);
  v14 = v12 + v13;
  if ( v12 != v12 + v13 )
  {
    v15 = 0;
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)a1;
        v17 = *(_QWORD *)(*(_QWORD *)a1 + 224LL);
        v18 = *(void (**)())(*(_QWORD *)v17 + 120LL);
        v58 = v15;
        v59[0] = "Type unit ";
        v60 = &v58;
        v61 = 2819;
        if ( v18 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v18)(v17, v59, 1);
          v16 = *(_QWORD *)a1;
        }
        v19 = *(_BYTE *)(v12 + 8);
        if ( v19 )
          break;
        sub_31F0D70(v16, *(_QWORD *)v12, 0);
LABEL_13:
        v12 += 16;
        ++v15;
        if ( v14 == v12 )
          goto LABEL_20;
      }
      if ( *((_BYTE *)a1 + 312) )
      {
        if ( v19 != 1 )
          goto LABEL_38;
        sub_31DCA30(v16, *(_QWORD *)v12);
        goto LABEL_13;
      }
      if ( v19 != 1 )
        goto LABEL_38;
      v20 = *(_QWORD *)v12;
      v12 += 16;
      ++v15;
      sub_31F0F00(v16, v20);
    }
    while ( v14 != v12 );
  }
LABEL_20:
  v21 = *((_QWORD *)a1 + 1);
  v22 = *(_QWORD **)(v21 + 184);
  v23 = *(_QWORD **)(v21 + 192);
  if ( v22 != v23 )
  {
    v24 = 0;
    v25 = 1;
    do
    {
      v27 = *(_QWORD *)a1;
      v28 = *(_QWORD *)(*(_QWORD *)a1 + 224LL);
      v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
      v58 = v24;
      v59[0] = "Bucket ";
      v60 = &v58;
      v61 = 2819;
      if ( v29 != nullsub_98 )
      {
        v56 = v22;
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v29)(v28, v59, 1);
        v27 = *(_QWORD *)a1;
        v22 = v56;
      }
      v26 = 0;
      v55 = v22;
      if ( v22[1] != *v22 )
        v26 = v25;
      ++v24;
      sub_31DCA10(v27, v26);
      v22 = v55 + 3;
      v25 += (__int64)(v55[1] - *v55) >> 3;
    }
    while ( v23 != v55 + 3 );
  }
  sub_3722350((__int64)a1);
  sub_3722B80((__int64 *)a1);
  sub_3722650((__int64)a1);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 224LL) + 208LL))(
    *(_QWORD *)(*(_QWORD *)a1 + 224LL),
    *((_QWORD *)a1 + 36),
    0);
  v30 = (__int64 *)*((_QWORD *)a1 + 10);
  v53 = &v30[*((unsigned int *)a1 + 22)];
  if ( v30 != v53 )
  {
    v57 = (__int64 *)*((_QWORD *)a1 + 10);
    do
    {
      v31 = *(_QWORD *)a1;
      v32 = *(_QWORD *)(*(_QWORD *)a1 + 224LL);
      v33 = *v57;
      v34 = *(void (**)())(*(_QWORD *)v32 + 120LL);
      v59[0] = "Abbrev code";
      v61 = 259;
      if ( v34 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v34)(v32, v59, 1);
        v31 = *(_QWORD *)a1;
      }
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v31 + 424LL))(
        v31,
        *(unsigned int *)(v33 + 12),
        0,
        0);
      v54 = *(_QWORD *)(*(_QWORD *)a1 + 224LL);
      v35 = *(void (**)())(*(_QWORD *)v54 + 120LL);
      v36 = sub_E02B90(*(_DWORD *)(v33 + 8));
      v37 = v54;
      v61 = 261;
      v59[0] = v36;
      v59[1] = v38;
      if ( v35 != nullsub_98 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v35)(v54, v59, 1);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)a1 + 424LL))(
        *(_QWORD *)a1,
        *(unsigned int *)(v33 + 8),
        0,
        0,
        v37);
      v39 = *(unsigned int **)(v33 + 16);
      v40 = &v39[2 * *(unsigned int *)(v33 + 24)];
      while ( v40 != v39 )
      {
        v41 = *(_QWORD *)a1;
        v42 = *v39;
        v39 += 2;
        v43 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(**(_QWORD **)a1 + 424LL);
        v44 = sub_E0CB90(v42);
        v43(v41, *(v39 - 2), v44, 0);
        v45 = *(_QWORD *)a1;
        v46 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(**(_QWORD **)a1 + 424LL);
        v47 = sub_E06AB0(*((unsigned __int16 *)v39 - 2));
        v46(v45, *((unsigned __int16 *)v39 - 2), v47, 0);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD, const char *, _QWORD))(**(_QWORD **)a1 + 424LL))(
        *(_QWORD *)a1,
        0,
        "End of abbrev",
        0);
      (*(void (__fastcall **)(_QWORD, _QWORD, const char *, _QWORD))(**(_QWORD **)a1 + 424LL))(
        *(_QWORD *)a1,
        0,
        "End of abbrev",
        0);
      ++v57;
    }
    while ( v53 != v57 );
  }
  (*(void (__fastcall **)(_QWORD, _QWORD, const char *, _QWORD))(**(_QWORD **)a1 + 424LL))(
    *(_QWORD *)a1,
    0,
    "End of abbrev list",
    0);
  v48 = *((_QWORD *)a1 + 37);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 224LL) + 208LL))(
    *(_QWORD *)(*(_QWORD *)a1 + 224LL),
    v48,
    0);
  sub_3725D40(a1, v48, v49, v50, v51);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 224LL) + 608LL))(
    *(_QWORD *)(*(_QWORD *)a1 + 224LL),
    2,
    0,
    1,
    0);
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 224LL) + 208LL))(
           *(_QWORD *)(*(_QWORD *)a1 + 224LL),
           *((_QWORD *)a1 + 35),
           0);
}
