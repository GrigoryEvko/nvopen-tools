// Function: sub_11E0B50
// Address: 0x11e0b50
//
__int64 __fastcall sub_11E0B50(__int64 a1, __int64 a2, unsigned int **a3)
{
  int v4; // edx
  unsigned __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdi
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 v12; // rdi
  int v13; // eax
  bool v14; // al
  bool v15; // al
  __int64 v16; // rcx
  _QWORD *v17; // r9
  char *v18; // rdi
  _QWORD *v19; // rsi
  __int64 v20; // r14
  __int64 v22; // rax
  _BYTE *v23; // rax
  __int64 **v24; // r14
  _BYTE *v25; // rax
  __int64 v26; // rdi
  __int64 (__fastcall *v27)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  _BYTE *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // r14
  __int64 v40; // rax
  char v41; // al
  _QWORD *v42; // rax
  __int64 v43; // r13
  __int64 v44; // r14
  unsigned int *v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned int *v48; // rdi
  __int64 **v49; // r14
  __int64 v50; // rdi
  __int64 (__fastcall *v51)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v52; // r9
  __int64 v53; // rax
  unsigned int *v54; // rdi
  __int64 v55; // rax
  _BYTE *v56; // rax
  unsigned int *v57; // rdi
  __int64 v58; // rax
  _BYTE *v59; // rax
  __int64 v60; // rax
  unsigned int *v61; // rdi
  __int64 v62; // r13
  __int64 v63; // rax
  _BYTE *v64; // rax
  unsigned int *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r15
  __int64 v69; // r13
  unsigned int *v70; // rbx
  __int64 v71; // rdx
  unsigned int v72; // esi
  unsigned int *v73; // rbx
  __int64 v74; // r15
  __int64 v75; // rdx
  unsigned int v76; // esi
  _BYTE *v77; // [rsp+10h] [rbp-D0h]
  __int64 v78; // [rsp+10h] [rbp-D0h]
  __int64 v79; // [rsp+10h] [rbp-D0h]
  __int64 v80; // [rsp+18h] [rbp-C8h]
  __int64 v81; // [rsp+20h] [rbp-C0h]
  _QWORD *v82; // [rsp+20h] [rbp-C0h]
  __int64 v83; // [rsp+20h] [rbp-C0h]
  char v84; // [rsp+20h] [rbp-C0h]
  char *v85; // [rsp+28h] [rbp-B8h]
  __int64 v86; // [rsp+28h] [rbp-B8h]
  __int64 v87; // [rsp+30h] [rbp-B0h]
  __int64 v88; // [rsp+38h] [rbp-A8h]
  char *v89; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v90; // [rsp+48h] [rbp-98h]
  _BYTE *v91[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v92; // [rsp+70h] [rbp-70h]
  int v93[8]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v94; // [rsp+A0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = *(_QWORD *)(a1 + 16);
  v93[0] = 0;
  v6 = v4 & 0x7FFFFFF;
  v87 = *(_QWORD *)(a2 - 32 * v6);
  v7 = *(_QWORD *)(a2 + 32 * (2 - v6));
  sub_11DAA90(a2, v93, 1, v7, v5);
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v7 != 17 )
  {
    v22 = sub_AD6530(v8, (__int64)v93);
    v16 = 0;
    v88 = v22;
    goto LABEL_8;
  }
  v10 = sub_AD6530(v8, (__int64)v93);
  v11 = *(_DWORD *)(v7 + 32);
  v12 = v7 + 24;
  v88 = v10;
  if ( v11 <= 0x40 )
  {
    v14 = *(_QWORD *)(v7 + 24) == 0;
  }
  else
  {
    v13 = sub_C444A0(v12);
    v12 = v7 + 24;
    v14 = v11 == v13;
  }
  if ( v14 )
    return v88;
  if ( v11 <= 0x40 )
    v15 = *(_QWORD *)(v7 + 24) == 1;
  else
    v15 = v11 - 1 == (unsigned int)sub_C444A0(v12);
  v16 = v7;
  if ( v15 )
  {
    v37 = sub_BCB2B0(a3[9]);
    v38 = (__int64)a3[6];
    v39 = v37;
    v92 = 259;
    v91[0] = "memrchr.char0";
    v40 = sub_AA4E30(v38);
    v41 = sub_AE5020(v40, v39);
    v94 = 257;
    v84 = v41;
    v42 = sub_BD2C40(80, unk_3F10A14);
    v43 = (__int64)v42;
    if ( v42 )
      sub_B4D190((__int64)v42, v39, v87, (__int64)v93, 0, v84, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, _BYTE **, unsigned int *, unsigned int *))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v43,
      v91,
      a3[7],
      a3[8]);
    if ( *a3 != &(*a3)[4 * *((unsigned int *)a3 + 2)] )
    {
      v44 = (__int64)&(*a3)[4 * *((unsigned int *)a3 + 2)];
      v45 = *a3;
      do
      {
        v46 = *((_QWORD *)v45 + 1);
        v47 = *v45;
        v45 += 4;
        sub_B99FD0(v43, v47, v46);
      }
      while ( (unsigned int *)v44 != v45 );
    }
    v48 = a3[9];
    v92 = 257;
    v49 = (__int64 **)sub_BCB2B0(v48);
    if ( v49 == *(__int64 ***)(v9 + 8) )
    {
      v52 = (_BYTE *)v9;
      goto LABEL_45;
    }
    v50 = (__int64)a3[10];
    v51 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v50 + 120LL);
    if ( v51 == sub_920130 )
    {
      if ( *(_BYTE *)v9 > 0x15u )
        goto LABEL_56;
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v52 = (_BYTE *)sub_ADAB70(38, v9, v49, 0);
      else
        v52 = (_BYTE *)sub_AA93C0(0x26u, v9, (__int64)v49);
    }
    else
    {
      v52 = (_BYTE *)v51(v50, 38u, (_BYTE *)v9, (__int64)v49);
    }
    if ( v52 )
    {
LABEL_45:
      *(_QWORD *)v93 = "memrchr.char0cmp";
      v94 = 259;
      v53 = sub_92B530(a3, 0x20u, v43, v52, (__int64)v93);
      v94 = 259;
      *(_QWORD *)v93 = "memrchr.sel";
      return sub_B36550(a3, v53, v87, v88, (__int64)v93, 0);
    }
LABEL_56:
    v94 = 257;
    v86 = sub_B51D30(38, v9, (__int64)v49, (__int64)v93, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, _BYTE **, unsigned int *, unsigned int *))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v86,
      v91,
      a3[7],
      a3[8]);
    v52 = (_BYTE *)v86;
    if ( *a3 != &(*a3)[4 * *((unsigned int *)a3 + 2)] )
    {
      v73 = *a3;
      v74 = (__int64)&(*a3)[4 * *((unsigned int *)a3 + 2)];
      do
      {
        v75 = *((_QWORD *)v73 + 1);
        v76 = *v73;
        v73 += 4;
        sub_B99FD0(v86, v76, v75);
      }
      while ( (unsigned int *)v74 != v73 );
      v52 = (_BYTE *)v86;
    }
    goto LABEL_45;
  }
LABEL_8:
  v81 = v16;
  v89 = 0;
  v90 = 0;
  if ( !(unsigned __int8)sub_98B0F0(v87, &v89, 0) )
    return 0;
  if ( !v90 )
    return v88;
  if ( v81 )
  {
    v17 = *(_QWORD **)(v81 + 24);
    if ( *(_DWORD *)(v81 + 32) > 0x40u )
      v17 = (_QWORD *)*v17;
    if ( v90 < v17 )
      return 0;
  }
  else
  {
    v17 = v90;
  }
  v18 = v89;
  if ( *(_BYTE *)v9 != 17 )
  {
LABEL_26:
    v90 = v17;
    if ( sub_C93580(&v89, *v18, 0) == -1 )
    {
      v83 = *(_QWORD *)(v7 + 8);
      v24 = (__int64 **)sub_BCB2B0(a3[9]);
      v94 = 257;
      v25 = (_BYTE *)sub_AD64C0(v83, 0, 0);
      v80 = sub_92B530(a3, 0x21u, v7, v25, (__int64)v93);
      v92 = 257;
      if ( v24 == *(__int64 ***)(v9 + 8) )
      {
        v28 = v9;
        goto LABEL_33;
      }
      v26 = (__int64)a3[10];
      v27 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v26 + 120LL);
      if ( v27 == sub_920130 )
      {
        if ( *(_BYTE *)v9 > 0x15u )
          goto LABEL_52;
        if ( (unsigned __int8)sub_AC4810(0x26u) )
          v28 = sub_ADAB70(38, v9, v24, 0);
        else
          v28 = sub_AA93C0(0x26u, v9, (__int64)v24);
      }
      else
      {
        v28 = v27(v26, 38u, (_BYTE *)v9, (__int64)v24);
      }
      if ( v28 )
      {
LABEL_33:
        v77 = (_BYTE *)v28;
        v94 = 257;
        v29 = sub_AD64C0((__int64)v24, *v89, 0);
        v30 = sub_92B530(a3, 0x20u, v29, v77, (__int64)v93);
        v94 = 257;
        v31 = v30;
        v32 = sub_AD6530(*(_QWORD *)(v30 + 8), 32);
        v33 = sub_B36550(a3, v80, v31, v32, (__int64)v93, 0);
        v94 = 257;
        v34 = v33;
        v35 = (_BYTE *)sub_AD64C0(v83, 1, 0);
        v91[0] = (_BYTE *)sub_929DE0(a3, (_BYTE *)v7, v35, (__int64)v93, 0, 0);
        *(_QWORD *)v93 = "memrchr.ptr_plus";
        v94 = 259;
        v36 = sub_921130(a3, (__int64)v24, v87, v91, 1, (__int64)v93, 3u);
        v94 = 259;
        *(_QWORD *)v93 = "memrchr.sel";
        return sub_B36550(a3, v34, v36, v88, (__int64)v93, 0);
      }
LABEL_52:
      v94 = 257;
      v78 = sub_B51D30(38, v9, (__int64)v24, (__int64)v93, 0, 0);
      (*(void (__fastcall **)(unsigned int *, __int64, _BYTE **, unsigned int *, unsigned int *))(*(_QWORD *)a3[11]
                                                                                                + 16LL))(
        a3[11],
        v78,
        v91,
        a3[7],
        a3[8]);
      v28 = v78;
      if ( *a3 != &(*a3)[4 * *((unsigned int *)a3 + 2)] )
      {
        v79 = v7;
        v68 = (__int64)&(*a3)[4 * *((unsigned int *)a3 + 2)];
        v69 = v28;
        v70 = *a3;
        do
        {
          v71 = *((_QWORD *)v70 + 1);
          v72 = *v70;
          v70 += 4;
          sub_B99FD0(v69, v72, v71);
        }
        while ( (unsigned int *)v68 != v70 );
        v28 = v69;
        v7 = v79;
      }
      goto LABEL_33;
    }
    return 0;
  }
  v19 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v19 = (_QWORD *)*v19;
  v20 = (__int64)v17;
  while ( v20 )
  {
    if ( (_BYTE)v19 == v89[--v20] )
    {
      if ( v81 )
      {
        v54 = a3[9];
        v94 = 257;
        v55 = sub_BCB2E0(v54);
        v56 = (_BYTE *)sub_ACD640(v55, v20, 0);
        v57 = a3[9];
        v91[0] = v56;
        v58 = sub_BCB2B0(v57);
        return sub_921130(a3, v58, v87, v91, 1, (__int64)v93, 3u);
      }
      v82 = v17;
      v85 = v89;
      v23 = memchr(v89, (char)v19, (size_t)v90);
      v18 = v85;
      v17 = v82;
      if ( v23 && v20 == v23 - v85 )
      {
        *(_QWORD *)v93 = "memrchr.cmp";
        v94 = 259;
        v59 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v7 + 8), v20, 0);
        v60 = sub_92B530(a3, 0x25u, v7, v59, (__int64)v93);
        v61 = a3[9];
        v62 = v60;
        v94 = 259;
        *(_QWORD *)v93 = "memrchr.ptr_plus";
        v63 = sub_BCB2E0(v61);
        v64 = (_BYTE *)sub_ACD640(v63, v20, 0);
        v65 = a3[9];
        v91[0] = v64;
        v66 = sub_BCB2B0(v65);
        v67 = sub_921130(a3, v66, v87, v91, 1, (__int64)v93, 3u);
        v94 = 259;
        *(_QWORD *)v93 = "memrchr.sel";
        return sub_B36550(a3, v62, v88, v67, (__int64)v93, 0);
      }
      goto LABEL_26;
    }
  }
  return v88;
}
