// Function: sub_1F9A430
// Address: 0x1f9a430
//
__int64 __fastcall sub_1F9A430(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // rsi
  char v15; // dl
  const void **v16; // rax
  bool v17; // zf
  __int64 *v19; // rax
  const void **v20; // r15
  __int16 v21; // ax
  unsigned int v22; // eax
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rcx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  int v32; // eax
  unsigned int v33; // eax
  unsigned int v34; // edx
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 *v42; // r8
  int v43; // eax
  __int64 v44; // r12
  int v45; // eax
  unsigned int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // r12
  unsigned int v51; // ebx
  unsigned int v52; // eax
  unsigned int v53; // ebx
  unsigned int v54; // ebx
  __int64 v55; // rax
  __int64 v56; // r12
  __int64 v57; // rax
  unsigned int v58; // eax
  __int64 *v59; // r12
  const void **v60; // rdx
  unsigned __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // r12
  __int64 v64; // rdx
  __int64 *v65; // r12
  __int64 v66; // rdx
  __int64 *v67; // r12
  __int64 v68; // rdx
  __int64 *v69; // r12
  const void **v70; // rdx
  int v71; // eax
  int v72; // eax
  int v73; // r12d
  unsigned __int64 v74; // rax
  int v75; // eax
  __int64 *v76; // rbx
  __int128 v77; // rax
  __int64 *v78; // rax
  __int64 v79; // rdx
  int v80; // [rsp+10h] [rbp-110h]
  const void **v81; // [rsp+10h] [rbp-110h]
  __int64 v82; // [rsp+18h] [rbp-108h]
  __int64 v83; // [rsp+18h] [rbp-108h]
  int v84; // [rsp+20h] [rbp-100h]
  __int128 *v85; // [rsp+20h] [rbp-100h]
  __int64 v86; // [rsp+20h] [rbp-100h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v88; // [rsp+28h] [rbp-F8h]
  int v89; // [rsp+30h] [rbp-F0h]
  unsigned int v90; // [rsp+30h] [rbp-F0h]
  __int64 *v91; // [rsp+30h] [rbp-F0h]
  unsigned int v92; // [rsp+38h] [rbp-E8h]
  __int64 *v93; // [rsp+38h] [rbp-E8h]
  unsigned int v94; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v95; // [rsp+38h] [rbp-E8h]
  unsigned int v96; // [rsp+40h] [rbp-E0h]
  unsigned int v97; // [rsp+40h] [rbp-E0h]
  __int128 v98; // [rsp+40h] [rbp-E0h]
  unsigned int v99; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v100; // [rsp+50h] [rbp-D0h]
  int v101; // [rsp+50h] [rbp-D0h]
  __int64 v102; // [rsp+50h] [rbp-D0h]
  __int128 v103; // [rsp+50h] [rbp-D0h]
  __int64 v104; // [rsp+50h] [rbp-D0h]
  unsigned int v105; // [rsp+60h] [rbp-C0h]
  unsigned int v106; // [rsp+60h] [rbp-C0h]
  __int128 v107; // [rsp+60h] [rbp-C0h]
  unsigned int v109; // [rsp+80h] [rbp-A0h] BYREF
  const void **v110; // [rsp+88h] [rbp-98h]
  __int64 v111; // [rsp+90h] [rbp-90h] BYREF
  int v112; // [rsp+98h] [rbp-88h]
  unsigned __int64 v113; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v114; // [rsp+A8h] [rbp-78h]
  __int64 *v115; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v116; // [rsp+B8h] [rbp-68h]
  __int64 v117[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v118[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v119[8]; // [rsp+E0h] [rbp-40h] BYREF

  v11 = a6;
  v13 = *(_QWORD *)(a4 + 40) + 16LL * a5;
  v14 = *(_QWORD *)(a6 + 72);
  v15 = *(_BYTE *)v13;
  v16 = *(const void ***)(v13 + 8);
  LOBYTE(v109) = v15;
  v110 = v16;
  v111 = v14;
  if ( v14 )
    sub_1623A60((__int64)&v111, v14, 2);
  v17 = *(_WORD *)(a2 + 24) == 48;
  v112 = *(_DWORD *)(v11 + 64);
  if ( v17 || *(_WORD *)(a4 + 24) == 48 )
  {
    v11 = sub_1D38BB0(*a1, 0, (__int64)&v111, v109, v110, 0, a7, a8, a9, 0);
    goto LABEL_6;
  }
  v19 = sub_1F85670((__int64)a1, 1u, a2, a3, a4, a5, a7, a8, a9, (__int64)&v111);
  v20 = (const void **)v19;
  if ( v19 )
  {
    v11 = (__int64)v19;
    goto LABEL_6;
  }
  v21 = *(_WORD *)(a2 + 24);
  if ( v21 != 52 )
  {
LABEL_12:
    if ( v21 != 124 )
      goto LABEL_13;
    if ( !sub_1D18C00(a2, 1, a3) )
      goto LABEL_13;
    v43 = *(unsigned __int16 *)(a4 + 24);
    if ( v43 != 32 && v43 != 10 )
      goto LABEL_13;
    v44 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
    v45 = *(unsigned __int16 *)(v44 + 24);
    if ( v45 != 10 && v45 != 32 )
      goto LABEL_13;
    v46 = (_BYTE)v109 ? sub_1F6C8D0(v109) : sub_1F58D40((__int64)&v109);
    v47 = *(_QWORD *)(v44 + 88);
    v102 = *(_DWORD *)(v47 + 32) <= 0x40u ? *(_QWORD *)(v47 + 24) : **(_QWORD **)(v47 + 24);
    if ( !(_DWORD)v102 )
      goto LABEL_13;
    v48 = *(_QWORD *)(a4 + 88);
    v93 = (__int64 *)(v48 + 24);
    if ( *(_DWORD *)(v48 + 32) > 0x40u )
    {
      v99 = v46;
      v72 = sub_16A58F0((__int64)v93);
      v46 = v99;
      v89 = v72;
    }
    else
    {
      _RBX = ~*(_QWORD *)(v48 + 24);
      if ( *(_QWORD *)(v48 + 24) == -1 )
        LODWORD(_RBX) = 64;
      else
        __asm { tzcnt   rbx, rbx }
      v89 = _RBX;
    }
    v51 = v46 >> 1;
    v97 = v46 >> 1;
    if ( v46 >> 1 == 32 )
    {
      LOBYTE(v52) = 5;
    }
    else
    {
      if ( v51 > 0x20 )
      {
        LOBYTE(v52) = 6;
        if ( v97 == 64 )
          goto LABEL_68;
        if ( v97 == 128 )
        {
          LOBYTE(v52) = 7;
          goto LABEL_68;
        }
        goto LABEL_85;
      }
      LOBYTE(v52) = 3;
      if ( v51 != 8 )
      {
        LOBYTE(v52) = 4;
        if ( v51 != 16 )
        {
          LOBYTE(v52) = 2;
          if ( v51 != 1 )
          {
LABEL_85:
            v52 = sub_1F58CC0(*(_QWORD **)(*a1 + 48), v97);
            v105 = v52;
            v20 = v70;
          }
        }
      }
    }
LABEL_68:
    v53 = v105;
    LOBYTE(v53) = v52;
    v106 = v53;
    v54 = *(_DWORD *)(v48 + 32);
    if ( v54 > 0x40 )
    {
      v73 = sub_16A58F0((__int64)v93);
      if ( !v73 || v54 != v73 + (unsigned int)sub_16A57B0((__int64)v93) )
        goto LABEL_13;
    }
    else
    {
      v55 = *(_QWORD *)(v48 + 24);
      if ( !v55 || (v55 & (v55 + 1)) != 0 )
        goto LABEL_13;
    }
    if ( v97 >= v89 + (int)v102
      && (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, const void **, _QWORD, const void **))(*(_QWORD *)a1[1]
                                                                                                  + 928LL))(
           a1[1],
           v109,
           v110,
           v106,
           v20)
      && (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, const void **))(*(_QWORD *)a1[1] + 1136LL))(
           a1[1],
           118,
           v106,
           v20)
      && (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, const void **))(*(_QWORD *)a1[1] + 1136LL))(
           a1[1],
           124,
           v106,
           v20)
      && (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, const void **, _QWORD, const void **))(*(_QWORD *)a1[1]
                                                                                                  + 800LL))(
           a1[1],
           v109,
           v110,
           v106,
           v20)
      && (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, const void **, _QWORD, const void **))(*(_QWORD *)a1[1]
                                                                                                  + 824LL))(
           a1[1],
           v106,
           v20,
           v109,
           v110) )
    {
      v56 = a1[1];
      v57 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32));
      v58 = sub_1F40B60(v56, v106, (__int64)v20, v57, 1);
      v59 = (__int64 *)*a1;
      v81 = v60;
      v90 = v58;
      v85 = *(__int128 **)(a2 + 32);
      sub_1F80610((__int64)v119, a2);
      v86 = sub_1D309E0(
              v59,
              145,
              (__int64)v119,
              v106,
              v20,
              0,
              *(double *)a7.m128i_i64,
              a8,
              *(double *)a9.m128i_i64,
              *v85);
      v88 = v61;
      sub_17CD270(v119);
      v83 = *a1;
      sub_1F80610((__int64)v119, a2);
      sub_16A5A50((__int64)v118, v93, v97);
      *(_QWORD *)&v98 = sub_1D38970(v83, (__int64)v118, (__int64)v119, v106, v20, 0, a7, a8, a9, 0);
      *((_QWORD *)&v98 + 1) = v62;
      sub_135E100(v118);
      sub_17CD270(v119);
      v63 = *a1;
      sub_1F80610((__int64)v119, a2);
      *(_QWORD *)&v103 = sub_1D38BB0(v63, (unsigned int)v102, (__int64)v119, v90, v81, 0, a7, a8, a9, 0);
      *((_QWORD *)&v103 + 1) = v64;
      sub_17CD270(v119);
      v65 = (__int64 *)*a1;
      sub_1F80610((__int64)v119, a2);
      *(_QWORD *)&v103 = sub_1D332F0(
                           v65,
                           124,
                           (__int64)v119,
                           v106,
                           v20,
                           0,
                           *(double *)a7.m128i_i64,
                           a8,
                           a9,
                           v86,
                           v88,
                           v103);
      *((_QWORD *)&v103 + 1) = v66;
      sub_17CD270(v119);
      v67 = (__int64 *)*a1;
      sub_1F80610((__int64)v119, a2);
      *(_QWORD *)&v107 = sub_1D332F0(
                           v67,
                           118,
                           (__int64)v119,
                           v106,
                           v20,
                           0,
                           *(double *)a7.m128i_i64,
                           a8,
                           a9,
                           v103,
                           *((unsigned __int64 *)&v103 + 1),
                           v98);
      *((_QWORD *)&v107 + 1) = v68;
      sub_17CD270(v119);
      v69 = (__int64 *)*a1;
      sub_1F80610((__int64)v119, a2);
      v11 = sub_1D309E0(
              v69,
              143,
              (__int64)v119,
              v109,
              v110,
              0,
              *(double *)a7.m128i_i64,
              a8,
              *(double *)a9.m128i_i64,
              v107);
      sub_17CD270(v119);
      goto LABEL_6;
    }
LABEL_13:
    v11 = 0;
    goto LABEL_6;
  }
  if ( *(_WORD *)(a4 + 24) != 124 )
    goto LABEL_13;
  v22 = (_BYTE)v109 ? sub_1F6C8D0(v109) : sub_1F58D40((__int64)&v109);
  if ( v22 > 0x40 )
    goto LABEL_13;
  v23 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  v24 = *(unsigned __int16 *)(v23 + 24);
  if ( v24 != 32 && v24 != 10 )
    goto LABEL_13;
  v25 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL);
  v26 = *(unsigned __int16 *)(v25 + 24);
  if ( v26 != 10 && v26 != 32 )
    goto LABEL_13;
  v27 = *(_QWORD *)(v23 + 88);
  v114 = *(_DWORD *)(v27 + 32);
  if ( v114 > 0x40 )
  {
    v104 = v25;
    sub_16A4FD0((__int64)&v113, (const void **)(v27 + 24));
    v25 = v104;
  }
  else
  {
    v113 = *(_QWORD *)(v27 + 24);
  }
  v28 = *(_QWORD *)(v25 + 88);
  v96 = *(_DWORD *)(v28 + 32);
  v116 = v96;
  if ( v96 > 0x40 )
  {
    sub_16A4FD0((__int64)&v115, (const void **)(v28 + 24));
    v96 = v116;
  }
  else
  {
    v115 = *(__int64 **)(v28 + 24);
  }
  v29 = v114;
  v100 = v113;
  v30 = 1LL << ((unsigned __int8)v114 - 1);
  if ( v114 > 0x40 )
  {
    v94 = v114;
    if ( (*(_QWORD *)(v113 + 8LL * ((v114 - 1) >> 6)) & v30) != 0 )
      v32 = sub_16A5810((__int64)&v113);
    else
      v32 = sub_16A57B0((__int64)&v113);
    v29 = v94;
  }
  else if ( (v30 & v113) != 0 )
  {
    if ( v113 << (64 - (unsigned __int8)v114) == -1 )
    {
      v32 = 64;
    }
    else
    {
      _BitScanReverse64(&v31, ~(v113 << (64 - (unsigned __int8)v114)));
      v32 = v31 ^ 0x3F;
    }
  }
  else
  {
    if ( v113 )
    {
      _BitScanReverse64(&v74, v113);
      v75 = v74 ^ 0x3F;
    }
    else
    {
      v75 = 64;
    }
    v32 = v114 + v75 - 64;
  }
  if ( v29 + 1 - v32 > 0x40 )
    goto LABEL_45;
  v92 = v29;
  v33 = (_BYTE)v109 ? sub_1F6C8D0(v109) : sub_1F58D40((__int64)&v109);
  v34 = v92;
  if ( v96 > 0x40 )
  {
    v91 = v115;
    v84 = v92;
    v95 = v33;
    v71 = sub_16A57B0((__int64)&v115);
    v34 = v84;
    v42 = v91;
    if ( v96 - v71 > 0x40 )
      goto LABEL_90;
    if ( v95 <= *v91 )
    {
LABEL_91:
      j_j___libc_free_0_0(v42);
LABEL_46:
      if ( v114 > 0x40 && v113 )
        j_j___libc_free_0_0(v113);
      v21 = *(_WORD *)(a2 + 24);
      goto LABEL_12;
    }
  }
  else if ( (unsigned __int64)v115 >= v33 )
  {
    goto LABEL_46;
  }
  v35 = v34 > 0x40 ? *(_QWORD *)v100 : (__int64)(v100 << (64 - (unsigned __int8)v34)) >> (64 - (unsigned __int8)v34);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1[1] + 760LL))(a1[1], v35) )
  {
LABEL_45:
    v42 = v115;
    if ( v116 <= 0x40 )
      goto LABEL_46;
LABEL_90:
    if ( !v42 )
      goto LABEL_46;
    goto LABEL_91;
  }
  v39 = (__int64)v115;
  if ( v116 > 0x40 )
    v39 = *v115;
  v101 = v39;
  v40 = sub_1D159A0((char *)&v109, v35, v39, v36, v37, v38, v80, v82, v84, v87);
  sub_171A350((__int64)v117, v40, v101);
  if ( !(unsigned __int8)sub_1D1F940(
                           *a1,
                           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
                           (__int64)v117,
                           0)
    || ((sub_1F6C9A0((__int64 *)&v113, v117), v114 > 0x40)
      ? (v41 = *(_QWORD *)v113)
      : (v41 = (__int64)(v113 << (64 - (unsigned __int8)v114)) >> (64 - (unsigned __int8)v114)),
        !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1[1] + 760LL))(a1[1], v41)) )
  {
    sub_135E100(v117);
    goto LABEL_45;
  }
  sub_1F80610((__int64)v118, a2);
  v76 = (__int64 *)*a1;
  *(_QWORD *)&v77 = sub_1D38970(*a1, (__int64)&v113, (__int64)&v111, v109, v110, 0, a7, a8, a9, 0);
  v78 = sub_1D332F0(
          v76,
          52,
          (__int64)v118,
          v109,
          v110,
          0,
          *(double *)a7.m128i_i64,
          a8,
          a9,
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          v77);
  v119[1] = v79;
  v119[0] = (__int64)v78;
  sub_1F994A0((__int64)a1, a2, v119, 1, 1);
  sub_17CD270(v118);
  sub_135E100(v117);
  sub_135E100((__int64 *)&v115);
  sub_135E100((__int64 *)&v113);
LABEL_6:
  if ( v111 )
    sub_161E7C0((__int64)&v111, v111);
  return v11;
}
