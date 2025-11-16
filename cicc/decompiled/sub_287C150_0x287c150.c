// Function: sub_287C150
// Address: 0x287c150
//
__int64 __fastcall sub_287C150(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r13
  __int64 v16; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned int v23; // r14d
  unsigned int v24; // eax
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r15
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  char v41; // bl
  unsigned __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rdi
  __int64 i; // rax
  __int64 v47; // r15
  unsigned __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r8
  _QWORD *v51; // rdx
  __int64 v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rbx
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int8 *v61; // rax
  _QWORD *v62; // r13
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rcx
  _QWORD *v66; // r13
  unsigned __int64 v67; // r14
  __int64 v68; // rsi
  __int64 v69; // r11
  __int64 v70; // rax
  __int64 v71; // rax
  _QWORD *v72; // rax
  _QWORD *v73; // r11
  _QWORD **v74; // rdx
  int v75; // ecx
  int v76; // eax
  __int64 *v77; // rax
  __int64 v78; // rax
  unsigned int *v79; // rbx
  unsigned int *v80; // r15
  __int64 v81; // r13
  __int64 v82; // rdx
  __int64 v83; // [rsp+0h] [rbp-530h]
  __int64 v84; // [rsp+8h] [rbp-528h]
  __int64 v85; // [rsp+18h] [rbp-518h]
  unsigned __int64 v86; // [rsp+20h] [rbp-510h]
  __int64 v87; // [rsp+28h] [rbp-508h]
  __int64 v89; // [rsp+40h] [rbp-4F0h]
  __int64 v90; // [rsp+48h] [rbp-4E8h]
  __int64 v91; // [rsp+48h] [rbp-4E8h]
  unsigned int v92; // [rsp+50h] [rbp-4E0h]
  char v93; // [rsp+58h] [rbp-4D8h]
  __int64 v94; // [rsp+60h] [rbp-4D0h]
  __int64 v95; // [rsp+68h] [rbp-4C8h]
  __int64 v96; // [rsp+68h] [rbp-4C8h]
  __int64 v97; // [rsp+68h] [rbp-4C8h]
  __int64 v98; // [rsp+68h] [rbp-4C8h]
  __int64 v99; // [rsp+68h] [rbp-4C8h]
  _QWORD *v102; // [rsp+78h] [rbp-4B8h]
  unsigned __int8 v103; // [rsp+87h] [rbp-4A9h]
  _BYTE *v104; // [rsp+88h] [rbp-4A8h]
  __int64 v105; // [rsp+98h] [rbp-498h] BYREF
  _QWORD v106[4]; // [rsp+A0h] [rbp-490h] BYREF
  char v107; // [rsp+C0h] [rbp-470h]
  char v108; // [rsp+C1h] [rbp-46Fh]
  _QWORD v109[4]; // [rsp+D0h] [rbp-460h] BYREF
  __int16 v110; // [rsp+F0h] [rbp-440h]
  unsigned int *v111; // [rsp+100h] [rbp-430h] BYREF
  __int64 v112; // [rsp+108h] [rbp-428h]
  _BYTE v113[32]; // [rsp+110h] [rbp-420h] BYREF
  __int64 v114; // [rsp+130h] [rbp-400h]
  __int64 v115; // [rsp+138h] [rbp-3F8h]
  __int64 v116; // [rsp+140h] [rbp-3F0h]
  __int64 v117; // [rsp+148h] [rbp-3E8h]
  void **v118; // [rsp+150h] [rbp-3E0h]
  void **v119; // [rsp+158h] [rbp-3D8h]
  __int64 v120; // [rsp+160h] [rbp-3D0h]
  int v121; // [rsp+168h] [rbp-3C8h]
  __int16 v122; // [rsp+16Ch] [rbp-3C4h]
  char v123; // [rsp+16Eh] [rbp-3C2h]
  __int64 v124; // [rsp+170h] [rbp-3C0h]
  __int64 v125; // [rsp+178h] [rbp-3B8h]
  void *v126; // [rsp+180h] [rbp-3B0h] BYREF
  void *v127; // [rsp+188h] [rbp-3A8h] BYREF
  _QWORD v128[116]; // [rsp+190h] [rbp-3A0h] BYREF

  if ( a6 )
  {
    v9 = sub_22077B0(0x2F8u);
    v10 = v9;
    if ( v9 )
    {
      *(_QWORD *)v9 = a6;
      *(_QWORD *)(v9 + 8) = v9 + 24;
      *(_QWORD *)(v9 + 16) = 0x1000000000LL;
      *(_QWORD *)(v9 + 416) = v9 + 440;
      *(_QWORD *)(v9 + 504) = v9 + 520;
      *(_QWORD *)(v9 + 512) = 0x800000000LL;
      *(_QWORD *)(v9 + 408) = 0;
      *(_QWORD *)(v9 + 424) = 8;
      *(_DWORD *)(v9 + 432) = 0;
      *(_BYTE *)(v9 + 436) = 1;
      *(_DWORD *)(v9 + 720) = 0;
      *(_QWORD *)(v9 + 728) = 0;
      *(_QWORD *)(v9 + 736) = v9 + 720;
      *(_QWORD *)(v9 + 744) = v9 + 720;
      *(_QWORD *)(v9 + 752) = 0;
    }
    if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a1 + 8) )
      goto LABEL_5;
  }
  else
  {
    if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a1 + 8) )
      return 0;
    v10 = 0;
  }
  if ( !(unsigned __int8)sub_D4B3D0(a1) || !(unsigned __int8)sub_DCFA10(a2, (char *)a1) )
    goto LABEL_5;
  v95 = sub_D47930(a1);
  v19 = *(_QWORD *)(v95 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 == v95 + 48 || !v19 || (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
    goto LABEL_123;
  if ( *(_BYTE *)(v19 - 24) != 31
    || (v20 = *(_DWORD *)(v19 - 20) & 0x7FFFFFF, (_DWORD)v20 == 1)
    || (v21 = *(_QWORD *)(v19 - 120), *(_BYTE *)v21 != 82)
    || (v22 = *(_QWORD *)(v21 + 16)) == 0
    || *(_QWORD *)(v22 + 8)
    || (v104 = *(_BYTE **)(v21 - 64), (unsigned __int8)(*v104 - 42) > 0x11u)
    || !(unsigned __int8)sub_D48480(a1, *(_QWORD *)(v21 - 32), v20, v18)
    || !sub_9913D0((__int64)v104, &v105, v106, v109)
    || *(_QWORD *)(v105 + 40) != **(_QWORD **)(a1 + 32)
    || (v103 = sub_F6EBA0(v105, v95, v21)) == 0 )
  {
LABEL_5:
    v103 = 0;
    goto LABEL_6;
  }
  v23 = 2 * LODWORD(qword_4F8C268[8]);
  v92 = 2 * LODWORD(qword_4F8C268[8]);
  v24 = sub_DBB070((__int64)a2, a1, 0);
  if ( v24 )
  {
    if ( v23 <= v24 )
      v24 = v23;
    v92 = v24;
  }
  else
  {
    v128[0] = sub_F6EC60(a1, 0);
    if ( BYTE4(v128[0]) )
    {
      v53 = v128[0];
      if ( v92 <= LODWORD(v128[0]) )
        v53 = v92;
      v92 = v53;
    }
  }
  v87 = sub_DCF3A0(a2, (char *)a1, 0);
  v25 = sub_AA4E30(**(_QWORD **)(a1 + 32));
  sub_27C1C30((__int64)v128, a2, v25, (__int64)"lsr_fold_term_cond", 1);
  v26 = sub_D4B130(a1);
  v86 = sub_986580(v26);
  v27 = sub_AA5930(**(_QWORD **)(a1 + 32));
  v94 = 0;
  v85 = v28;
  v93 = 0;
  v83 = a4;
  v90 = 0;
  v84 = v10;
  v29 = v27;
  while ( v85 != v29 )
  {
    if ( v105 != v29 && sub_D97040((__int64)a2, *(_QWORD *)(v29 + 8)) )
    {
      v30 = sub_DD8400((__int64)a2, v29);
      v34 = (__int64)v30;
      if ( *((_WORD *)v30 + 12) == 8 && v30[5] == 2 && (*((_BYTE *)v30 + 28) & 1) != 0 )
      {
        v35 = sub_D33D80(v30, (__int64)a2, v31, v32, v33);
        if ( (unsigned __int8)sub_DBE090((__int64)a2, v35) )
        {
          v36 = sub_DCC620(v34, a2);
          v111 = (unsigned int *)sub_DD0540((__int64)v36, v87, a2);
          if ( (unsigned __int8)sub_F80610((__int64)v128, (__int64)v111, v37, v38, v39, v40) )
          {
            v41 = sub_F6CE90((int)v128, (__int64 *)&v111, 1, a1, v92, v83, v86);
            if ( !v41 )
            {
              v42 = sub_986580(v95);
              if ( (unsigned __int8)sub_98EF90(v29, v42, a3, v43, v44) )
              {
                v45 = *(_QWORD *)(v29 - 8);
                for ( i = 0; (*(_DWORD *)(v29 + 4) & 0x7FFFFFF) != (_DWORD)i; ++i )
                {
                  if ( v95 == *(_QWORD *)(v45 + 32LL * *(unsigned int *)(v29 + 72) + 8 * i) )
                    goto LABEL_65;
                }
                LODWORD(i) = -1;
LABEL_65:
                v47 = *(_QWORD *)(v45 + 32LL * (unsigned int)i);
                v48 = sub_986580(v95);
                if ( (unsigned __int8)sub_98EF90(v47, v48, a3, v49, v50) )
                  goto LABEL_78;
                v51 = (*(_BYTE *)(v47 + 7) & 0x40) != 0
                    ? *(_QWORD **)(v47 - 8)
                    : (_QWORD *)(v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF));
                if ( v29 == *v51 )
                {
                  v41 = sub_B44920(v47);
LABEL_78:
                  v93 = v41;
                  v90 = v29;
                  v94 = (__int64)v111;
                  goto LABEL_70;
                }
              }
            }
          }
        }
      }
    }
    if ( !v29 )
      BUG();
LABEL_70:
    v52 = *(_QWORD *)(v29 + 32);
    if ( !v52 )
      goto LABEL_123;
    v29 = 0;
    if ( *(_BYTE *)(v52 - 24) == 84 )
      v29 = v52 - 24;
  }
  v10 = v84;
  v54 = a1;
  if ( !v105 || !v90 )
  {
    sub_27C20B0((__int64)v128);
    goto LABEL_5;
  }
  sub_27C20B0((__int64)v128);
  v55 = sub_D4B130(a1);
  v56 = sub_D47930(a1);
  v57 = *(_QWORD *)(v90 - 8);
  v58 = v56;
  if ( (*(_DWORD *)(v90 + 4) & 0x7FFFFFF) != 0 )
  {
    v59 = 0;
    while ( v58 != *(_QWORD *)(v57 + 32LL * *(unsigned int *)(v90 + 72) + 8 * v59) )
    {
      if ( (*(_DWORD *)(v90 + 4) & 0x7FFFFFF) == (_DWORD)++v59 )
        goto LABEL_108;
    }
    v60 = 32 * v59;
  }
  else
  {
LABEL_108:
    v60 = 0x1FFFFFFFE0LL;
  }
  v61 = *(unsigned __int8 **)(v57 + v60);
  v96 = (__int64)v61;
  if ( v93 )
    sub_B44F30(v61);
  v62 = (_QWORD *)(v55 + 48);
  v63 = sub_AA4E30(**(_QWORD **)(a1 + 32));
  sub_27C1C30((__int64)v128, a2, v63, (__int64)"lsr_fold_term_cond", 1);
  v64 = *v62 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v64 == v62 )
  {
    v65 = 0;
    goto LABEL_92;
  }
  if ( !v64 )
    goto LABEL_123;
  v65 = v64 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v64 - 24) - 30 >= 0xB )
    v65 = 0;
LABEL_92:
  v66 = sub_F8DB90((__int64)v128, v94, *(_QWORD *)(v90 + 8), v65 + 24, 0);
  v67 = *(_QWORD *)(v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v67 == v58 + 48 )
LABEL_117:
    BUG();
  if ( !v67 )
LABEL_123:
    BUG();
  v91 = v67 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v67 - 24) - 30 > 0xA )
    goto LABEL_117;
  v102 = *(_QWORD **)(v67 - 120);
  v117 = sub_BD5C60(v91);
  v118 = &v126;
  v119 = &v127;
  v122 = 512;
  v112 = 0x200000000LL;
  v126 = &unk_49DA100;
  v111 = (unsigned int *)v113;
  LOWORD(v116) = 0;
  v127 = &unk_49DA0B0;
  v120 = 0;
  v121 = 0;
  v123 = 7;
  v124 = 0;
  v125 = 0;
  v114 = 0;
  v115 = 0;
  sub_D5F1F0((__int64)&v111, v91);
  v108 = 1;
  v68 = 32;
  v106[0] = "lsr_fold_term_cond.replaced_term_cond";
  v107 = 3;
  v69 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD *))*v118 + 7))(v118, 32, v96, v66);
  if ( !v69 )
  {
    v110 = 257;
    v72 = sub_BD2C40(72, unk_3F10FD0);
    v73 = v72;
    if ( v72 )
    {
      v89 = (__int64)v72;
      v74 = *(_QWORD ***)(v96 + 8);
      v75 = *((unsigned __int8 *)v74 + 8);
      if ( (unsigned int)(v75 - 17) > 1 )
      {
        v78 = sub_BCB2A0(*v74);
      }
      else
      {
        v76 = *((_DWORD *)v74 + 8);
        BYTE4(v105) = (_BYTE)v75 == 18;
        LODWORD(v105) = v76;
        v77 = (__int64 *)sub_BCB2A0(*v74);
        v78 = sub_BCE1B0(v77, v105);
      }
      sub_B523C0(v89, v78, 53, 32, v96, (__int64)v66, (__int64)v109, 0, 0, 0);
      v73 = (_QWORD *)v89;
    }
    v97 = (__int64)v73;
    v68 = (__int64)v73;
    (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, __int64, __int64))*v119 + 2))(v119, v73, v106, v115, v116);
    v69 = v97;
    if ( v111 != &v111[4 * (unsigned int)v112] )
    {
      v98 = v54;
      v79 = v111;
      v80 = &v111[4 * (unsigned int)v112];
      v81 = v69;
      do
      {
        v82 = *((_QWORD *)v79 + 1);
        v68 = *v79;
        v79 += 4;
        sub_B99FD0(v81, v68, v82);
      }
      while ( v80 != v79 );
      v54 = v98;
      v69 = v81;
    }
  }
  if ( *(_QWORD *)(v67 - 56) == **(_QWORD **)(v54 + 32) )
  {
    v99 = v69;
    sub_B4CC70(v91);
    v69 = v99;
  }
  if ( *(_QWORD *)(v67 - 120) )
  {
    v70 = *(_QWORD *)(v67 - 112);
    **(_QWORD **)(v67 - 104) = v70;
    if ( v70 )
      *(_QWORD *)(v70 + 16) = *(_QWORD *)(v67 - 104);
  }
  *(_QWORD *)(v67 - 120) = v69;
  if ( v69 )
  {
    v71 = *(_QWORD *)(v69 + 16);
    *(_QWORD *)(v67 - 112) = v71;
    if ( v71 )
      *(_QWORD *)(v71 + 16) = v67 - 112;
    *(_QWORD *)(v67 - 104) = v69 + 16;
    *(_QWORD *)(v69 + 16) = v67 - 120;
  }
  sub_F82360((__int64)v128, v68);
  sub_B43D60(v102);
  sub_F39260(**(_QWORD **)(v54 + 32), a5, v84);
  nullsub_61();
  v126 = &unk_49DA100;
  nullsub_63();
  if ( v111 != (unsigned int *)v113 )
    _libc_free((unsigned __int64)v111);
  sub_27C20B0((__int64)v128);
LABEL_6:
  if ( v10 )
  {
    sub_287BB70(*(_QWORD **)(v10 + 728));
    v11 = *(_QWORD **)(v10 + 504);
    v12 = &v11[3 * *(unsigned int *)(v10 + 512)];
    if ( v11 != v12 )
    {
      do
      {
        v13 = *(v12 - 1);
        v12 -= 3;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v12);
      }
      while ( v11 != v12 );
      v12 = *(_QWORD **)(v10 + 504);
    }
    if ( v12 != (_QWORD *)(v10 + 520) )
      _libc_free((unsigned __int64)v12);
    if ( !*(_BYTE *)(v10 + 436) )
      _libc_free(*(_QWORD *)(v10 + 416));
    v14 = *(_QWORD **)(v10 + 8);
    v15 = &v14[3 * *(unsigned int *)(v10 + 16)];
    if ( v14 != v15 )
    {
      do
      {
        v16 = *(v15 - 1);
        v15 -= 3;
        if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
          sub_BD60C0(v15);
      }
      while ( v14 != v15 );
      v15 = *(_QWORD **)(v10 + 8);
    }
    if ( v15 != (_QWORD *)(v10 + 24) )
      _libc_free((unsigned __int64)v15);
    j_j___libc_free_0(v10);
  }
  return v103;
}
