// Function: sub_925930
// Address: 0x925930
//
__int64 __fastcall sub_925930(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  unsigned int **v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 i; // rax
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rax
  bool v26; // zf
  _BYTE *v27; // r13
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rcx
  int v31; // eax
  int v32; // edx
  int v33; // eax
  __int64 v34; // rax
  unsigned int v35; // r14d
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rcx
  _BYTE *v39; // r14
  __int64 v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  int v45; // ebx
  int v46; // r8d
  __int64 v47; // rdi
  __int64 v48; // r15
  __int64 v49; // rax
  char v50; // al
  __int16 v51; // cx
  __int64 v52; // rax
  __int64 v53; // r9
  __int64 v54; // r13
  unsigned int *v55; // r15
  unsigned int *v56; // rbx
  __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rdi
  __int64 (__fastcall *v60)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v61; // rbx
  __int64 v62; // rdi
  __int64 (__fastcall *v63)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v64; // rax
  __int64 k; // rax
  __int64 v66; // r13
  __int64 v67; // r13
  __int64 v68; // r12
  _BYTE *v69; // rax
  _BYTE *v71; // rax
  unsigned int *v72; // rax
  unsigned int *v73; // r13
  unsigned int *v74; // rbx
  __int64 v75; // rdx
  __int64 v76; // rsi
  int v77; // edx
  __int64 v78; // rax
  __int64 v79; // rdx
  unsigned int *v80; // r13
  unsigned int *v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rsi
  unsigned __int8 v84; // al
  __int64 v85; // r11
  __int64 v86; // rax
  _BYTE *v87; // rbx
  __int64 v88; // rax
  __int64 v89; // rdi
  _BYTE *v90; // r11
  __int64 (__fastcall *v91)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  __int64 v92; // rax
  unsigned int *v93; // rax
  unsigned int *v94; // r15
  unsigned int *j; // rbx
  __int64 v96; // rdx
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v99; // [rsp-8h] [rbp-128h]
  __int64 v101; // [rsp+18h] [rbp-108h]
  __int64 v102; // [rsp+28h] [rbp-F8h]
  __int64 v103; // [rsp+38h] [rbp-E8h]
  int v104; // [rsp+40h] [rbp-E0h]
  __int64 v105; // [rsp+40h] [rbp-E0h]
  _BYTE *v106; // [rsp+40h] [rbp-E0h]
  _BYTE *v107; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v108; // [rsp+48h] [rbp-D8h]
  __int64 v109; // [rsp+50h] [rbp-D0h]
  __int64 v110; // [rsp+58h] [rbp-C8h]
  __int16 v111; // [rsp+66h] [rbp-BAh]
  __int64 v112; // [rsp+70h] [rbp-B0h]
  __int64 v113; // [rsp+78h] [rbp-A8h]
  _BYTE *v114; // [rsp+88h] [rbp-98h] BYREF
  unsigned int v115[8]; // [rsp+90h] [rbp-90h] BYREF
  char v116; // [rsp+B0h] [rbp-70h]
  char v117; // [rsp+B1h] [rbp-6Fh]
  _QWORD v118[4]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v119; // [rsp+E0h] [rbp-40h]

  v15 = (unsigned int **)(a2 + 48);
  if ( (unsigned __int8)sub_920010(a12, a11, (unsigned int)a10) )
  {
    v19 = sub_9232A0(a2, a3, 0, v16, v17, v18, a7, a8, a9, a10, a11, a12, a13);
    for ( i = *(_QWORD *)(a11 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v21 = *(_QWORD *)(i + 128);
    v22 = *(unsigned __int8 *)(a11 + 137);
    v119 = 259;
    v23 = 8 * v21 - v22;
    v24 = v23 - *(unsigned __int8 *)(a11 + 136) - 8 * (*(_QWORD *)(a11 + 128) % v21);
    v118[0] = "highclear";
    v25 = sub_920C00(v15, v19, v24, (__int64)v118, 0, 0);
    v26 = (*(_BYTE *)(a11 + 144) & 8) == 0;
    HIBYTE(v119) = 1;
    v27 = (_BYTE *)v25;
    if ( v26 )
    {
      v118[0] = "zeroext";
      LOBYTE(v119) = 3;
      v28 = sub_920DA0(v15, v25, v23, (__int64)v118, 0);
    }
    else
    {
      LOBYTE(v119) = 3;
      v118[0] = "signext";
      v71 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v25 + 8), v23, 0);
      v28 = sub_920F70(v15, v27, v71, (__int64)v118, 0);
    }
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v28;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    return a1;
  }
  v29 = *(_QWORD *)(a2 + 40);
  v30 = *(_QWORD *)(a11 + 128);
  v31 = *(unsigned __int8 *)(a11 + 137) + *(unsigned __int8 *)(a11 + 136);
  v32 = v31 + 6;
  v33 = v31 - 1;
  v110 = v30;
  if ( v33 < 0 )
    v33 = v32;
  v34 = v33 >> 3;
  v108 = v30 + v34;
  v109 = __CFADD__(v30, v34);
  v35 = *(_DWORD *)(*(_QWORD *)(a8 + 8) + 8LL);
  v118[0] = "bf.base.i8ptr";
  v119 = 259;
  v36 = sub_BCB2B0(v29);
  v37 = sub_BCE760(v36, v35 >> 8);
  v101 = sub_920710(v15, 0x31u, a8, v37, (__int64)v118, 0, v115[0], 0);
  v103 = sub_91A3A0(*(_QWORD *)(a2 + 32) + 8LL, *(_QWORD *)(a11 + 120), *(_QWORD *)(a2 + 32), v38);
  v39 = (_BYTE *)sub_AD64C0(v103, 0, 0);
  v102 = *(unsigned __int8 *)(a11 + 137);
  if ( !v109 )
  {
    v112 = a2 + 48;
    v113 = v110;
    while ( 1 )
    {
      v40 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
      v41 = (_BYTE *)sub_ACD640(v40, v113, 0);
      v42 = *(_QWORD *)(a2 + 40);
      v119 = 257;
      v114 = v41;
      v43 = sub_BCB2B0(v42);
      v44 = sub_921130((unsigned int **)v112, v43, v101, &v114, 1, (__int64)v118, 0);
      v45 = v44;
      if ( unk_4D0463C && sub_90AA40(*(_QWORD *)(a2 + 32), v44) )
        v46 = 1;
      else
        v46 = a13 & 1;
      v47 = *(_QWORD *)(a2 + 40);
      v104 = v46;
      v117 = 1;
      *(_QWORD *)v115 = "bf.curbyte";
      v116 = 3;
      v48 = sub_BCB2B0(v47);
      v49 = sub_AA4E30(*(_QWORD *)(a2 + 96));
      v50 = sub_AE5020(v49, v48);
      HIBYTE(v51) = HIBYTE(v111);
      v119 = 257;
      LOBYTE(v51) = v50;
      v111 = v51;
      v52 = sub_BD2C40(80, unk_3F10A14);
      v54 = v52;
      if ( v52 )
      {
        sub_B4D190(v52, v48, v45, (unsigned int)v118, v104, (unsigned __int8)v111, 0, 0);
        v53 = v99;
      }
      (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v54,
        v115,
        *(_QWORD *)(v112 + 56),
        *(_QWORD *)(v112 + 64),
        v53);
      v55 = *(unsigned int **)(a2 + 48);
      v56 = &v55[4 * *(unsigned int *)(a2 + 56)];
      while ( v56 != v55 )
      {
        v57 = *((_QWORD *)v55 + 1);
        v58 = *v55;
        v55 += 4;
        sub_B99FD0(v54, v58, v57);
      }
      if ( v108 == v113 )
        break;
LABEL_18:
      if ( v110 == v113 )
      {
        v84 = *(_BYTE *)(a11 + 136);
        if ( v84 )
        {
          v119 = 257;
          v54 = sub_920DA0((unsigned int **)v112, v54, v84, (__int64)v118, 0);
        }
      }
      v117 = 1;
      *(_QWORD *)v115 = "bf.byte_zext";
      v116 = 3;
      if ( v103 == *(_QWORD *)(v54 + 8) )
      {
        v61 = (_BYTE *)v54;
        goto LABEL_25;
      }
      v59 = *(_QWORD *)(a2 + 128);
      v60 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v59 + 120LL);
      if ( v60 != sub_920130 )
      {
        v61 = (_BYTE *)v60(v59, 39u, (_BYTE *)v54, v103);
        goto LABEL_24;
      }
      if ( *(_BYTE *)v54 <= 0x15u )
      {
        if ( (unsigned __int8)sub_AC4810(39) )
          v61 = (_BYTE *)sub_ADAB70(39, v54, v103, 0);
        else
          v61 = (_BYTE *)sub_AA93C0(39, v54, v103);
LABEL_24:
        if ( v61 )
          goto LABEL_25;
      }
      v119 = 257;
      v78 = sub_BD2C40(72, unk_3F10A14);
      v61 = (_BYTE *)v78;
      if ( v78 )
        sub_B515B0(v78, v54, v103, v118, 0, 0);
      (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v61,
        v115,
        *(_QWORD *)(v112 + 56),
        *(_QWORD *)(v112 + 64));
      v79 = 4LL * *(unsigned int *)(a2 + 56);
      v80 = *(unsigned int **)(a2 + 48);
      v81 = &v80[v79];
      while ( v81 != v80 )
      {
        v82 = *((_QWORD *)v80 + 1);
        v83 = *v80;
        v80 += 4;
        sub_B99FD0(v61, v83, v82);
      }
LABEL_25:
      if ( v109 )
      {
        v118[0] = "bf.position";
        v119 = 259;
        v61 = (_BYTE *)sub_920C00((unsigned int **)v112, (__int64)v61, v109, (__int64)v118, 0, 0);
      }
      v62 = *(_QWORD *)(a2 + 128);
      v117 = 1;
      *(_QWORD *)v115 = "bf.merge";
      v116 = 3;
      v63 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v62 + 16LL);
      if ( v63 == sub_9202E0 )
      {
        if ( *v61 > 0x15u || *v39 > 0x15u )
          goto LABEL_45;
        if ( (unsigned __int8)sub_AC47B0(29) )
          v64 = sub_AD5570(29, v61, v39, 0, 0);
        else
          v64 = sub_AABE40(29, v61, v39);
      }
      else
      {
        v64 = v63(v62, 29u, v61, v39);
      }
      if ( v64 )
      {
        v39 = (_BYTE *)v64;
LABEL_34:
        if ( v110 == v113 )
          goto LABEL_48;
        goto LABEL_35;
      }
LABEL_45:
      v119 = 257;
      v39 = (_BYTE *)sub_B504D0(29, v61, v39, v118, 0, 0);
      (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v39,
        v115,
        *(_QWORD *)(v112 + 56),
        *(_QWORD *)(v112 + 64));
      v72 = *(unsigned int **)(a2 + 48);
      v73 = &v72[4 * *(unsigned int *)(a2 + 56)];
      v74 = v72;
      if ( v72 == v73 )
        goto LABEL_34;
      do
      {
        v75 = *((_QWORD *)v74 + 1);
        v76 = *v74;
        v74 += 4;
        sub_B99FD0(v39, v76, v75);
      }
      while ( v73 != v74 );
      if ( v110 == v113 )
      {
LABEL_48:
        v77 = *(unsigned __int8 *)(a11 + 136);
        v102 -= 8 - v77;
        v109 = 8 - v77;
        goto LABEL_36;
      }
LABEL_35:
      v109 += 8;
      v102 -= 8;
LABEL_36:
      if ( v108 < ++v113 )
      {
        v15 = (unsigned int **)v112;
        goto LABEL_38;
      }
    }
    v85 = 8 - v102;
    if ( v110 == v108 )
      v85 = 8LL - *(unsigned __int8 *)(a11 + 136) - v102;
    v119 = 257;
    v105 = v85;
    v86 = sub_920C00((unsigned int **)v112, v54, v85, (__int64)v118, 0, 0);
    v117 = 1;
    v87 = (_BYTE *)v86;
    v116 = 3;
    *(_QWORD *)v115 = "bf.end.highclear";
    v88 = sub_AD64C0(*(_QWORD *)(v86 + 8), v105, 0);
    v89 = *(_QWORD *)(a2 + 128);
    v90 = (_BYTE *)v88;
    v91 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v89 + 24LL);
    if ( v91 == sub_920250 )
    {
      if ( *v87 > 0x15u || *v90 > 0x15u )
      {
LABEL_68:
        v119 = 257;
        v54 = sub_B504D0(26, v87, v90, v118, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v54,
          v115,
          *(_QWORD *)(v112 + 56),
          *(_QWORD *)(v112 + 64));
        v93 = *(unsigned int **)(a2 + 48);
        v94 = &v93[4 * *(unsigned int *)(a2 + 56)];
        for ( j = v93; v94 != j; j += 4 )
        {
          v96 = *((_QWORD *)j + 1);
          v97 = *j;
          sub_B99FD0(v54, v97, v96);
        }
        goto LABEL_18;
      }
      v106 = v90;
      if ( (unsigned __int8)sub_AC47B0(26) )
        v92 = sub_AD5570(26, v87, v106, 0, 0);
      else
        v92 = sub_AABE40(26, v87, v106);
      v90 = v106;
      v54 = v92;
    }
    else
    {
      v107 = v90;
      v98 = v91(v89, 26u, v87, v90, 0);
      v90 = v107;
      v54 = v98;
    }
    if ( v54 )
      goto LABEL_18;
    goto LABEL_68;
  }
LABEL_38:
  if ( (*(_BYTE *)(a11 + 144) & 8) != 0 )
  {
    for ( k = *(_QWORD *)(a11 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v66 = 8LL * *(_QWORD *)(k + 128);
    v119 = 259;
    v67 = v66 - *(unsigned __int8 *)(a11 + 137);
    v118[0] = "bf.highclear";
    v68 = sub_920C00(v15, (__int64)v39, v67, (__int64)v118, 0, 0);
    v119 = 259;
    v118[0] = "bf.finalval";
    v69 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v68 + 8), v67, 0);
    v39 = (_BYTE *)sub_920F70(v15, (_BYTE *)v68, v69, (__int64)v118, 0);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v39;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
