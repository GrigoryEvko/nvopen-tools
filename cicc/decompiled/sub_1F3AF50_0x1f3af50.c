// Function: sub_1F3AF50
// Address: 0x1f3af50
//
unsigned __int64 __fastcall sub_1F3AF50(
        _QWORD *a1,
        __int64 a2,
        unsigned int *a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6)
{
  __int64 v6; // r11
  char *v7; // r14
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // r10
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned __int16 v13; // bx
  bool v14; // al
  __int64 (*v15)(void); // rax
  __int64 v16; // r13
  __int16 v17; // dx
  __int64 (*v18)(); // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  _DWORD *v21; // rsi
  _DWORD *v22; // r15
  int v23; // r14d
  int v24; // r13d
  __int64 v25; // rdx
  int v26; // eax
  _QWORD *v27; // r9
  __int64 v28; // rdx
  int v29; // edx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  unsigned __int64 v33; // r14
  __int64 i; // r14
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r10
  __int64 v39; // rdx
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // r12d
  __int64 v44; // rax
  __int64 v46; // rax
  __int64 v47; // rcx
  unsigned int *v48; // rbx
  char *v49; // r14
  __int64 v50; // r15
  __int64 v51; // r12
  __int64 v52; // rsi
  unsigned int v53; // eax
  __int64 v54; // rax
  __int64 (*v55)(); // rax
  unsigned int v56; // edx
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v60; // [rsp+16h] [rbp-BAh]
  __int64 v62; // [rsp+20h] [rbp-B0h]
  __int64 v63; // [rsp+28h] [rbp-A8h]
  unsigned __int16 v66; // [rsp+40h] [rbp-90h]
  unsigned int v67; // [rsp+44h] [rbp-8Ch]
  __int64 v68; // [rsp+48h] [rbp-88h]
  __int64 v70; // [rsp+48h] [rbp-88h]
  __int64 v71; // [rsp+48h] [rbp-88h]
  unsigned int v73; // [rsp+50h] [rbp-80h]
  int v74; // [rsp+50h] [rbp-80h]
  __int64 v75; // [rsp+50h] [rbp-80h]
  int v76; // [rsp+50h] [rbp-80h]
  __int64 v77; // [rsp+50h] [rbp-80h]
  char *v78; // [rsp+58h] [rbp-78h]
  _QWORD v79[4]; // [rsp+60h] [rbp-70h] BYREF
  __int128 v80; // [rsp+80h] [rbp-50h] BYREF
  __int64 v81; // [rsp+90h] [rbp-40h]

  v6 = a5;
  v7 = (char *)&a3[a4];
  v8 = a2;
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 56LL);
  v78 = (char *)a3;
  v10 = *(_QWORD *)(v9 + 56);
  v11 = *(_QWORD *)(v9 + 16);
  v62 = *(_QWORD *)(a2 + 24);
  if ( a3 == (unsigned int *)v7 )
  {
    v55 = *(__int64 (**)())(*(_QWORD *)v11 + 112LL);
    if ( v55 == sub_1D00B10 )
    {
      v13 = 0;
      v66 = 0;
      goto LABEL_52;
    }
    v77 = *(_QWORD *)(v9 + 56);
    v13 = 0;
    v58 = v55();
    v10 = v77;
    v6 = a5;
    v66 = 0;
    v63 = v58;
LABEL_32:
    if ( v78 != v7 )
    {
      v12 = *(_QWORD *)(v8 + 32);
      goto LABEL_34;
    }
LABEL_52:
    LODWORD(v16) = 0;
    goto LABEL_7;
  }
  v12 = *(_QWORD *)(a2 + 32);
  v13 = 0;
  do
  {
    v14 = (*(_BYTE *)(v12 + 40LL * *a3++ + 3) & 0x10) != 0;
    v13 |= v14 + 1;
  }
  while ( v7 != (char *)a3 );
  v66 = v13;
  v15 = *(__int64 (**)(void))(*(_QWORD *)v11 + 112LL);
  if ( v15 != sub_1D00B10 )
  {
    v75 = *(_QWORD *)(v9 + 56);
    v46 = v15();
    v10 = v75;
    v6 = a5;
    v63 = v46;
    if ( (v13 & 2) != 0 )
      goto LABEL_6;
    goto LABEL_32;
  }
  v63 = 0;
  if ( (v13 & 2) == 0 )
  {
LABEL_34:
    v16 = 0;
    v47 = v8;
    v60 = v13;
    v48 = (unsigned int *)v7;
    v59 = v9;
    v49 = v78;
    v50 = v10;
    while ( 1 )
    {
      v51 = *(_QWORD *)(*(_QWORD *)(v50 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v50 + 32) + v6) + 8);
      v52 = (*(_DWORD *)(v12 + 40LL * *(unsigned int *)v49) >> 8) & 0xFFF;
      if ( (_DWORD)v52 && (v67 = v6, v70 = v47, v53 = sub_38D70C0(v63 + 8, v52), v47 = v70, v6 = v67, v53) )
      {
        if ( (v53 & 7) == 0 )
          v51 = v53 >> 3;
        if ( v16 < v51 )
          v16 = v51;
        v49 += 4;
        if ( v48 == (unsigned int *)v49 )
          goto LABEL_46;
      }
      else
      {
        if ( v16 < v51 )
          v16 = v51;
        v49 += 4;
        if ( v48 == (unsigned int *)v49 )
        {
LABEL_46:
          v10 = v50;
          v13 = v60;
          v8 = v47;
          v9 = v59;
          goto LABEL_7;
        }
      }
      v12 = *(_QWORD *)(v47 + 32);
    }
  }
LABEL_6:
  v16 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v6) + 8);
LABEL_7:
  v17 = **(_WORD **)(v8 + 16);
  if ( (v17 & 0xFFFB) == 0x13 || v17 == 21 )
  {
    v68 = v10;
    v74 = v6;
    v35 = sub_1F3A290(v9, v8, v78, a4, v6, a1);
    LODWORD(v6) = v74;
    v33 = (unsigned __int64)v35;
    if ( !v35 )
      goto LABEL_48;
    sub_1DD5BA0((__int64 *)(v62 + 16), (__int64)v35);
    v36 = *(_QWORD *)v8;
    v37 = *(_QWORD *)v33;
    *(_QWORD *)(v33 + 8) = v8;
    LODWORD(v6) = v74;
    v38 = v68;
    v36 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v33 = v36 | v37 & 7;
    *(_QWORD *)(v36 + 8) = v33;
    *(_QWORD *)v8 = v33 | *(_QWORD *)v8 & 7LL;
    goto LABEL_29;
  }
  v18 = *(__int64 (**)())(*a1 + 504LL);
  if ( v18 != sub_1F39480 )
  {
    v71 = v10;
    v76 = v6;
    v54 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, char *, __int64, __int64, __int64, __int64))v18)(
            a1,
            v9,
            v8,
            v78,
            a4,
            v8,
            v6,
            a6);
    LODWORD(v6) = v76;
    v38 = v71;
    v33 = v54;
    if ( !v54 )
    {
LABEL_48:
      if ( **(_WORD **)(v8 + 16) != 15 )
        return 0;
      goto LABEL_11;
    }
LABEL_29:
    v39 = *(_QWORD *)(v8 + 56);
    v40 = *(_BYTE *)(v8 + 49);
    v79[2] = 0;
    *(_QWORD *)(v33 + 56) = v39;
    *(_BYTE *)(v33 + 49) = v40;
    v41 = (unsigned int)(*(_DWORD *)(v38 + 32) + v6);
    v79[1] = 0;
    v42 = *(_QWORD *)(v38 + 8);
    v79[0] = 0;
    v43 = *(_DWORD *)(v42 + 40 * v41 + 16);
    sub_1E341E0((__int64)&v80, v9, v6, 0);
    v44 = sub_1E0B8E0(v9, v66, v16, v43, (int)v79, 0, v80, v81, 1u, 0, 0);
    sub_1E15C90(v33, v9, v44);
    return v33;
  }
  if ( v17 != 15 )
    return 0;
LABEL_11:
  if ( a4 != 1 )
    return 0;
  v73 = v6;
  if ( *(_DWORD *)(v8 + 40) != 2 )
    return 0;
  v19 = *(_QWORD *)(v8 + 32);
  v20 = *(unsigned int *)v78;
  v21 = (_DWORD *)(v19 + 40 * v20);
  v22 = (_DWORD *)(v19 + 40LL * (unsigned int)(1 - v20));
  if ( (*v21 & 0xFFF00) != 0 || (*v22 & 0xFFF00) != 0 )
    return 0;
  v23 = v21[2];
  v24 = v22[2];
  v25 = *(_QWORD *)(*(_QWORD *)(sub_1E15F70(v8) + 40) + 24LL);
  v26 = v22[2];
  v27 = (_QWORD *)(*(_QWORD *)(v25 + 16LL * (v23 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v26 <= 0 )
  {
    v56 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v25 + 16LL * (v24 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) + 24LL);
    v57 = *(_DWORD *)(v27[1] + 4 * ((unsigned __int64)(unsigned __int16)v56 >> 5));
    if ( _bittest(&v57, v56) )
      goto LABEL_18;
    return 0;
  }
  v28 = (unsigned int)v26 >> 3;
  if ( (unsigned int)v28 >= *(unsigned __int16 *)(*v27 + 22LL) )
    return 0;
  v29 = *(unsigned __int8 *)(*(_QWORD *)(*v27 + 8LL) + v28);
  if ( !_bittest(&v29, v22[2] & 7) )
    return 0;
LABEL_18:
  v30 = *(_QWORD *)(v8 + 32) + 40LL * (unsigned int)(1 - *(_DWORD *)v78);
  v31 = *(unsigned int *)(v30 + 8);
  v32 = *a1;
  if ( v13 == 2 )
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD *))(v32 + 408))(
      a1,
      v62,
      v8,
      v31,
      (*(_BYTE *)(v30 + 3) >> 6) & ((*(_BYTE *)(v30 + 3) >> 4) ^ 1) & 1,
      v73,
      v27);
  else
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(v32 + 416))(a1, v62, v8, v31, v73);
  v33 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v33 )
    BUG();
  if ( (*(_QWORD *)v33 & 4) == 0 && (*(_BYTE *)(v33 + 46) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v33; ; i = *(_QWORD *)v33 )
    {
      v33 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v33 + 46) & 4) == 0 )
        break;
    }
  }
  return v33;
}
