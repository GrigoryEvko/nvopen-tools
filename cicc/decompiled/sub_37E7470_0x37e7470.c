// Function: sub_37E7470
// Address: 0x37e7470
//
void __fastcall sub_37E7470(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 (*v7)(); // r10
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 (*v27)(); // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  unsigned __int64 *v30; // rcx
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // r13
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r11
  __int64 v45; // r13
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rdi
  _QWORD *v67; // r10
  __int64 (*v68)(); // rax
  char v69; // al
  _QWORD *v70; // [rsp+8h] [rbp-118h]
  _QWORD *v71; // [rsp+10h] [rbp-110h]
  __int64 v72; // [rsp+10h] [rbp-110h]
  __int64 v73; // [rsp+10h] [rbp-110h]
  __int64 v74; // [rsp+10h] [rbp-110h]
  __int64 v75; // [rsp+10h] [rbp-110h]
  __int64 v76; // [rsp+18h] [rbp-108h]
  __int64 v77; // [rsp+18h] [rbp-108h]
  __int64 v78; // [rsp+18h] [rbp-108h]
  __int64 v79; // [rsp+18h] [rbp-108h]
  __int64 v80; // [rsp+18h] [rbp-108h]
  __int64 v81; // [rsp+18h] [rbp-108h]
  int v82; // [rsp+24h] [rbp-FCh] BYREF
  __int64 v83; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v84; // [rsp+30h] [rbp-F0h] BYREF
  int *v85; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v86; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v87; // [rsp+48h] [rbp-D8h]
  _BYTE v88[208]; // [rsp+50h] [rbp-D0h] BYREF

  v4 = *(_QWORD *)(a2 + 56);
  v83 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v83, v4, 1);
  v5 = a1[65];
  v6 = *(_QWORD *)(a2 + 24);
  v86 = v88;
  v84 = 0;
  v85 = 0;
  v87 = 0x400000000LL;
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 344LL);
  v8 = 0;
  if ( v7 != sub_2DB1AE0 )
  {
    ((void (__fastcall *)(__int64, __int64, __int64 *, int **, _BYTE **, _QWORD))v7)(v5, v6, &v84, &v85, &v86, 0);
    v8 = v84;
  }
  v9 = *(_DWORD *)(v8 + 252);
  v10 = *(_DWORD *)(v8 + 256);
  if ( (*(_DWORD *)(v6 + 256) != v10 || *(_DWORD *)(v6 + 252) != v9) && v9 == unk_501EB38 && v10 == unk_501EB3C )
  {
    v28 = a1[43];
    if ( v28 )
    {
      v79 = sub_37E6EE0((__int64)a1, v28, *(_QWORD *)(v6 + 16));
      if ( (unsigned __int8)sub_37E6CB0(a1, a2, (int *)v79) )
      {
        v54 = *(int *)(v79 + 24);
        v55 = a1[25];
        v56 = a1[65];
        v82 = 0;
        v74 = v55 + 8 * v54;
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, int *))(*(_QWORD *)v56 + 368LL))(
          v56,
          v79,
          v84,
          0,
          0,
          0,
          &v83,
          &v82);
        *(_DWORD *)(v74 + 4) += v82;
        sub_2E33690(v6, v84, v79);
        v70 = (_QWORD *)v79;
        sub_2E33F80(v79, v84, -1, v57, v58, v59);
        v60 = *(int *)(v6 + 24);
        v61 = a1[25];
        v62 = a1[65];
        v82 = 0;
        v81 = v61 + 8 * v60;
        (*(void (__fastcall **)(__int64, __int64, int *))(*(_QWORD *)v62 + 360LL))(v62, v6, &v82);
        *(_DWORD *)(v81 + 4) -= v82;
        v63 = *(int *)(v6 + 24);
        v64 = a1[25];
        v65 = a1[65];
        v82 = 0;
        v75 = v64 + 8 * v63;
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, int *, _BYTE *, _QWORD, __int64 *, int *))(*(_QWORD *)v65 + 368LL))(
          v65,
          v6,
          v70,
          v85,
          v86,
          (unsigned int)v87,
          &v83,
          &v82);
        *(_DWORD *)(v75 + 4) += v82;
        a1[43] = v70;
        sub_37E6D50((__int64)a1, *v70 & 0xFFFFFFFFFFFFFFF8LL, v70[1]);
        v66 = a1[64];
        v67 = v70;
        v68 = *(__int64 (**)())(*(_QWORD *)v66 + 528LL);
        if ( v68 == sub_2FF52D0 || (v69 = ((__int64 (__fastcall *)(__int64, _QWORD))v68)(v66, a1[63]), v67 = v70, v69) )
          sub_3509790(a1 + 55, v67);
        goto LABEL_13;
      }
      v29 = v79;
      v71 = (_QWORD *)v79;
      *(_BYTE *)(a1[43] + 261LL) = *(_BYTE *)(v79 + 261);
      v80 = a1[63] + 320LL;
      sub_2E31020(v80, v29);
      v30 = (unsigned __int64 *)v71[1];
      v31 = *v71 & 0xFFFFFFFFFFFFFFF8LL;
      *v30 = v31 | *v30 & 7;
      *(_QWORD *)(v31 + 8) = v30;
      *v71 &= 7uLL;
      v71[1] = 0;
      sub_2E79D60(v80, v71);
    }
  }
  v11 = a1[65];
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 880LL);
  if ( v12 != sub_2DB1B20 && !((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v12)(v11, &v86) )
  {
    if ( v85 )
    {
      if ( (unsigned __int8)sub_37E6CB0(a1, a2, v85) )
      {
        v49 = *(int *)(v6 + 24);
        v50 = a1[25];
        v82 = 0;
        v51 = v50 + 8 * v49;
        (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)a1[65] + 360LL))(a1[65], v6);
        *(_DWORD *)(v51 + 4) = *(_DWORD *)(v51 + 4);
        v52 = *(int *)(v6 + 24);
        v53 = a1[25];
        v82 = 0;
        v48 = v53 + 8 * v52;
        (*(void (__fastcall **)(_QWORD, __int64, int *, __int64, _BYTE *, _QWORD))(*(_QWORD *)a1[65] + 368LL))(
          a1[65],
          v6,
          v85,
          v84,
          v86,
          (unsigned int)v87);
        goto LABEL_31;
      }
      if ( v85 )
      {
        v32 = sub_37E6EE0((__int64)a1, v6, *(_QWORD *)(v6 + 16));
        v33 = *(int *)(v32 + 24);
        v34 = (_QWORD *)v32;
        v35 = a1[65];
        v82 = 0;
        v72 = a1[25] + 8 * v33;
        (*(void (__fastcall **)(__int64, __int64, int *, _QWORD, _QWORD, _QWORD, __int64 *, int *))(*(_QWORD *)v35
                                                                                                  + 368LL))(
          v35,
          v32,
          v85,
          0,
          0,
          0,
          &v83,
          &v82);
        *(_DWORD *)(v72 + 4) += v82;
        sub_2E33690(v6, (__int64)v85, (__int64)v34);
        sub_2E33F80((__int64)v34, (__int64)v85, -1, v36, v37, v38);
        sub_37E6D50((__int64)a1, *v34 & 0xFFFFFFFFFFFFFFF8LL, v34[1]);
        v39 = a1[64];
        v40 = *(__int64 (**)())(*(_QWORD *)v39 + 528LL);
        if ( v40 == sub_2FF52D0 || ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v40)(v39, a1[63]) )
          sub_3509790(a1 + 55, v34);
      }
    }
    v41 = *(int *)(v6 + 24);
    v42 = a1[25];
    v43 = a1[65];
    v44 = *(_QWORD *)(v6 + 8);
    v82 = 0;
    v45 = v42 + 8 * v41;
    v73 = v44;
    (*(void (__fastcall **)(__int64, __int64, int *))(*(_QWORD *)v43 + 360LL))(v43, v6, &v82);
    *(_DWORD *)(v45 + 4) -= v82;
    v46 = *(int *)(v6 + 24);
    v47 = a1[25];
    v82 = 0;
    v48 = v47 + 8 * v46;
    (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, _BYTE *, _QWORD))(*(_QWORD *)a1[65] + 368LL))(
      a1[65],
      v6,
      v73,
      v84,
      v86,
      (unsigned int)v87);
LABEL_31:
    *(_DWORD *)(v48 + 4) += v82;
    goto LABEL_13;
  }
  if ( !v85 )
    v85 = *(int **)(v6 + 8);
  v13 = sub_37E6EE0((__int64)a1, v6, *(_QWORD *)(v6 + 16));
  v14 = *(int *)(v13 + 24);
  v15 = (_QWORD *)v13;
  v16 = a1[65];
  v82 = 0;
  v76 = a1[25] + 8 * v14;
  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, int *))(*(_QWORD *)v16 + 368LL))(
    v16,
    v13,
    v84,
    0,
    0,
    0,
    &v83,
    &v82);
  *(_DWORD *)(v76 + 4) += v82;
  sub_2E33690(v6, v84, (__int64)v15);
  sub_2E33F80((__int64)v15, v84, -1, v17, v18, v19);
  v20 = *(int *)(v6 + 24);
  v21 = a1[25];
  v22 = a1[65];
  v82 = 0;
  v77 = v21 + 8 * v20;
  (*(void (__fastcall **)(__int64, __int64, int *))(*(_QWORD *)v22 + 360LL))(v22, v6, &v82);
  *(_DWORD *)(v77 + 4) -= v82;
  v23 = *(int *)(v6 + 24);
  v24 = a1[25];
  v25 = a1[65];
  v82 = 0;
  v78 = v24 + 8 * v23;
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, int *, _BYTE *, _QWORD, __int64 *, int *))(*(_QWORD *)v25 + 368LL))(
    v25,
    v6,
    v15,
    v85,
    v86,
    (unsigned int)v87,
    &v83,
    &v82);
  *(_DWORD *)(v78 + 4) += v82;
  sub_37E6D50((__int64)a1, *v15 & 0xFFFFFFFFFFFFFFF8LL, v15[1]);
  v26 = a1[64];
  v27 = *(__int64 (**)())(*(_QWORD *)v26 + 528LL);
  if ( v27 == sub_2FF52D0 || ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v27)(v26, a1[63]) )
    sub_3509790(a1 + 55, v15);
LABEL_13:
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v83 )
    sub_B91220((__int64)&v83, v83);
}
