// Function: sub_2B33D00
// Address: 0x2b33d00
//
__int64 __fastcall sub_2B33D00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 *v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 *v19; // rdi
  __int64 v20; // r11
  char v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // r13
  __int64 v26; // r12
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rcx
  __int64 v30; // r9
  __int64 *v31; // r15
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // r14d
  __int64 v37; // r11
  __int64 v38; // r9
  int v39; // edi
  unsigned int v40; // edx
  __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 v46; // r8
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // r9
  __int64 v50; // r8
  char v51; // cl
  __int64 v52; // rdi
  __int64 *v53; // r15
  __int64 v54; // rsi
  __int64 v55; // r9
  int v56; // r8d
  unsigned int v57; // esi
  __int64 v58; // rax
  __int64 v59; // r10
  __int64 v60; // rdx
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r9
  __int64 v64; // rdx
  unsigned __int64 v65; // r8
  __int64 v66; // rsi
  __int64 v67; // rcx
  __int64 v68; // rdi
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // r10
  __int64 v72; // rdx
  char v73; // di
  __int64 v74; // rax
  __int64 *v75; // r15
  __int64 v76; // r9
  int v77; // edi
  unsigned int v78; // edx
  __int64 v79; // rax
  __int64 v80; // r10
  __int64 v81; // rdx
  double v82; // xmm0_8
  unsigned __int8 *v83; // rax
  __int64 v84; // r9
  __int64 v85; // rdx
  unsigned __int64 v86; // r8
  __int64 v87; // rsi
  __int64 v88; // rcx
  __int64 v89; // rsi
  __int64 v90; // rdi
  unsigned int v91; // edx
  __int64 *v92; // rax
  __int64 v93; // r10
  __int64 v94; // r8
  char v95; // cl
  __int64 v96; // rdi
  __int64 v97; // rbx
  _BYTE *v98; // rax
  __int64 v99; // rax
  int v100; // eax
  int v101; // eax
  int v102; // eax
  __int64 v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rdi
  __int64 *v106; // rdi
  __int64 v107; // rax
  __int64 v108; // r12
  __int64 v109; // rax
  int v110; // eax
  int v111; // eax
  int v112; // eax
  int v113; // r13d
  __int64 v114; // rax
  __int64 v115; // rdx
  unsigned int *v116; // rbx
  __int64 v117; // r12
  __int64 v118; // rdx
  unsigned int v119; // esi
  unsigned int *v120; // rbx
  __int64 v121; // r13
  __int64 v122; // rdx
  unsigned int v123; // esi
  int v124; // r9d
  int v125; // r9d
  int v126; // edi
  int v127; // r11d
  int v128; // r11d
  __int64 v129; // [rsp+0h] [rbp-110h]
  __int64 v130; // [rsp+10h] [rbp-100h]
  __int64 v131; // [rsp+10h] [rbp-100h]
  char v132; // [rsp+10h] [rbp-100h]
  unsigned __int8 *v133; // [rsp+10h] [rbp-100h]
  int v134; // [rsp+18h] [rbp-F8h]
  __int64 v135; // [rsp+18h] [rbp-F8h]
  int v136; // [rsp+18h] [rbp-F8h]
  int v137; // [rsp+18h] [rbp-F8h]
  __int64 v138; // [rsp+20h] [rbp-F0h]
  int v139; // [rsp+20h] [rbp-F0h]
  unsigned int v141; // [rsp+30h] [rbp-E0h]
  _BYTE v142[32]; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v143; // [rsp+60h] [rbp-B0h]
  _BYTE v144[32]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v145; // [rsp+90h] [rbp-80h]
  __int64 *v146; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v147; // [rsp+A8h] [rbp-68h]
  _BYTE v148[16]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v149; // [rsp+C0h] [rbp-50h]

  v8 = a2;
  v11 = *(__int64 **)a4;
  v12 = *(_QWORD *)(v8 + 8);
  v13 = ***(__int64 ****)a4;
  v14 = *(unsigned int *)(**(_QWORD **)a4 + 8LL);
  v134 = *(_DWORD *)(**(_QWORD **)a4 + 8LL);
  if ( *(_QWORD *)(v12 + 24) == *(_QWORD *)(*v13 + 8) )
    goto LABEL_5;
  v15 = *(_QWORD *)(a4 + 3528);
  v16 = *(unsigned int *)(a4 + 3544);
  v149 = 257;
  v17 = *v11;
  if ( !(_DWORD)v16 )
    goto LABEL_69;
  v18 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v19 = (__int64 *)(v15 + 24LL * v18);
  v20 = *v19;
  if ( v17 != *v19 )
  {
    v126 = 1;
    while ( v20 != -4096 )
    {
      v18 = (v16 - 1) & (v126 + v18);
      v139 = v126 + 1;
      v19 = (__int64 *)(v15 + 24LL * v18);
      v20 = *v19;
      if ( v17 == *v19 )
        goto LABEL_4;
      v126 = v139;
    }
LABEL_69:
    v19 = (__int64 *)(v15 + 24 * v16);
  }
LABEL_4:
  v21 = *((_BYTE *)v19 + 16);
  v130 = v14;
  v138 = a1;
  v22 = sub_2B08680(*(_QWORD *)(*v13 + 8), *(_DWORD *)(v12 + 32));
  v23 = sub_921630((unsigned int **)a3, a2, v22, v21, (__int64)&v146);
  v14 = v130;
  a1 = v138;
  v8 = v23;
LABEL_5:
  switch ( *(_DWORD *)(a1 + 1576) )
  {
    case 0:
    case 2:
    case 0xB:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
      BUG();
    case 1:
      v53 = &v13[v14];
      v146 = (__int64 *)v148;
      v147 = 0x600000000LL;
      if ( v53 == v13 )
      {
        v24 = (__int64 *)v148;
        v54 = 0;
LABEL_7:
        v25 = sub_AD3730(v24, v54);
        v143 = 257;
        v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a3 + 80)
                                                                                           + 32LL))(
                *(_QWORD *)(a3 + 80),
                17,
                v8,
                v25,
                0,
                0);
        if ( !v26 )
        {
          v145 = 257;
          v26 = sub_B504D0(17, v8, v25, (__int64)v144, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v26,
            v142,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v120 = *(unsigned int **)a3;
          v121 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v121 )
          {
            do
            {
              v122 = *((_QWORD *)v120 + 1);
              v123 = *v120;
              v120 += 4;
              sub_B99FD0(v26, v123, v122);
            }
            while ( (unsigned int *)v121 != v120 );
          }
        }
        if ( v146 != (__int64 *)v148 )
          _libc_free((unsigned __int64)v146);
        return v26;
      }
      while ( 1 )
      {
        v66 = *(unsigned int *)(a6 + 24);
        v67 = *v13;
        v68 = *(_QWORD *)(a6 + 8);
        if ( (_DWORD)v66 )
        {
          v69 = (v66 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v70 = (__int64 *)(v68 + 16LL * v69);
          v71 = *v70;
          if ( v67 == *v70 )
            goto LABEL_43;
          v100 = 1;
          while ( v71 != -4096 )
          {
            v125 = v100 + 1;
            v69 = (v66 - 1) & (v100 + v69);
            v70 = (__int64 *)(v68 + 16LL * v69);
            v71 = *v70;
            if ( v67 == *v70 )
              goto LABEL_43;
            v100 = v125;
          }
        }
        v70 = (__int64 *)(v68 + 16 * v66);
LABEL_43:
        v72 = v70[1];
        v73 = *(_BYTE *)(a5 + 8) & 1;
        if ( v73 )
        {
          v55 = a5 + 16;
          v56 = 15;
        }
        else
        {
          v74 = *(unsigned int *)(a5 + 24);
          v55 = *(_QWORD *)(a5 + 16);
          if ( !(_DWORD)v74 )
            goto LABEL_81;
          v56 = v74 - 1;
        }
        v57 = v56 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v58 = v55 + 16LL * v57;
        v59 = *(_QWORD *)v58;
        if ( v72 != *(_QWORD *)v58 )
        {
          v110 = 1;
          while ( v59 != -4096 )
          {
            v127 = v110 + 1;
            v57 = v56 & (v110 + v57);
            v58 = v55 + 16LL * v57;
            v59 = *(_QWORD *)v58;
            if ( v72 == *(_QWORD *)v58 )
              goto LABEL_34;
            v110 = v127;
          }
          if ( v73 )
          {
            v104 = 256;
          }
          else
          {
            v74 = *(unsigned int *)(a5 + 24);
LABEL_81:
            v104 = 16 * v74;
          }
          v58 = v55 + v104;
        }
LABEL_34:
        v60 = 256;
        if ( !v73 )
          v60 = 16LL * *(unsigned int *)(a5 + 24);
        v61 = 0;
        if ( v58 != v55 + v60 )
          v61 = *(unsigned int *)(*(_QWORD *)(a5 + 272) + 16LL * *(unsigned int *)(v58 + 8) + 8);
        v62 = sub_AD64C0(*(_QWORD *)(v67 + 8), v61, 0);
        v64 = (unsigned int)v147;
        v65 = (unsigned int)v147 + 1LL;
        if ( v65 > HIDWORD(v147) )
        {
          v135 = v62;
          sub_C8D5F0((__int64)&v146, v148, (unsigned int)v147 + 1LL, 8u, v65, v63);
          v64 = (unsigned int)v147;
          v62 = v135;
        }
        ++v13;
        v146[v64] = v62;
        v54 = (unsigned int)(v147 + 1);
        LODWORD(v147) = v147 + 1;
        if ( v53 == v13 )
        {
          v24 = v146;
          goto LABEL_7;
        }
      }
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
      return v8;
    case 5:
      v131 = v14;
      v28 = *(unsigned int *)(*(_QWORD *)(v8 + 8) + 32LL);
      v146 = (__int64 *)v148;
      v147 = 0xC00000000LL;
      sub_11B1960((__int64)&v146, v28, -1, a4, a5, v14);
      v29 = (unsigned __int64)v146;
      v30 = v131;
      v31 = (__int64 *)((char *)v146 + 4 * (unsigned int)v147);
      if ( v146 != v31 )
      {
        v32 = 0;
        v33 = (4 * (unsigned __int64)(unsigned int)v147 - 4) >> 2;
        do
        {
          v34 = v32;
          *(_DWORD *)(v29 + 4 * v32) = v32;
          ++v32;
        }
        while ( v33 != v34 );
        v31 = v146;
      }
      if ( !v134 )
        goto LABEL_64;
      v129 = v8;
      v35 = 0;
      v36 = v134;
      v132 = 0;
      v37 = 4 * v30;
      while ( 1 )
      {
        v44 = *(unsigned int *)(a6 + 24);
        v45 = v13[(unsigned __int64)v35 / 4];
        v46 = *(_QWORD *)(a6 + 8);
        if ( (_DWORD)v44 )
        {
          v47 = (v44 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v48 = (__int64 *)(v46 + 16LL * v47);
          v49 = *v48;
          if ( v45 == *v48 )
            goto LABEL_27;
          v102 = 1;
          while ( v49 != -4096 )
          {
            v47 = (v44 - 1) & (v102 + v47);
            v136 = v102 + 1;
            v48 = (__int64 *)(v46 + 16LL * v47);
            v49 = *v48;
            if ( v45 == *v48 )
              goto LABEL_27;
            v102 = v136;
          }
        }
        v48 = (__int64 *)(v46 + 16 * v44);
LABEL_27:
        v50 = v48[1];
        v51 = *(_BYTE *)(a5 + 8) & 1;
        if ( v51 )
        {
          v38 = a5 + 16;
          v39 = 15;
        }
        else
        {
          v52 = *(unsigned int *)(a5 + 24);
          v38 = *(_QWORD *)(a5 + 16);
          if ( !(_DWORD)v52 )
            goto LABEL_78;
          v39 = v52 - 1;
        }
        v40 = v39 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
        v41 = v38 + 16LL * v40;
        v42 = *(_QWORD *)v41;
        if ( v50 != *(_QWORD *)v41 )
        {
          v112 = 1;
          while ( v42 != -4096 )
          {
            v40 = v39 & (v112 + v40);
            v137 = v112 + 1;
            v41 = v38 + 16LL * v40;
            v42 = *(_QWORD *)v41;
            if ( v50 == *(_QWORD *)v41 )
              goto LABEL_20;
            v112 = v137;
          }
          if ( v51 )
          {
            v103 = 256;
          }
          else
          {
            v52 = *(unsigned int *)(a5 + 24);
LABEL_78:
            v103 = 16 * v52;
          }
          v41 = v38 + v103;
        }
LABEL_20:
        v43 = 256;
        if ( !v51 )
          v43 = 16LL * *(unsigned int *)(a5 + 24);
        if ( v41 == v38 + v43 || (*(_BYTE *)(*(_QWORD *)(a5 + 272) + 16LL * *(unsigned int *)(v41 + 8) + 8) & 1) == 0 )
        {
          *(_DWORD *)((char *)v31 + v35) = v36;
          v31 = v146;
          v132 = 1;
        }
        v35 += 4;
        if ( v37 == v35 )
        {
          v8 = v129;
          if ( v132 )
          {
            v97 = (unsigned int)v147;
            v145 = 257;
            v98 = (_BYTE *)sub_AD6530(*(_QWORD *)(v129 + 8), v35);
            v99 = sub_A83CB0((unsigned int **)a3, (_BYTE *)v129, v98, (__int64)v31, v97, (__int64)v144);
            v31 = v146;
            v8 = v99;
          }
LABEL_64:
          if ( v31 != (__int64 *)v148 )
            _libc_free((unsigned __int64)v31);
          return v8;
        }
      }
    case 0xA:
      v75 = &v13[v14];
      v146 = (__int64 *)v148;
      v147 = 0x600000000LL;
      if ( v13 != v75 )
      {
        while ( 1 )
        {
          v88 = *(unsigned int *)(a6 + 24);
          v89 = *v13;
          v90 = *(_QWORD *)(a6 + 8);
          if ( (_DWORD)v88 )
          {
            v91 = (v88 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
            v92 = (__int64 *)(v90 + 16LL * v91);
            v93 = *v92;
            if ( v89 == *v92 )
              goto LABEL_59;
            v101 = 1;
            while ( v93 != -4096 )
            {
              v124 = v101 + 1;
              v91 = (v88 - 1) & (v101 + v91);
              v92 = (__int64 *)(v90 + 16LL * v91);
              v93 = *v92;
              if ( v89 == *v92 )
                goto LABEL_59;
              v101 = v124;
            }
          }
          v92 = (__int64 *)(v90 + 16 * v88);
LABEL_59:
          v94 = v92[1];
          v95 = *(_BYTE *)(a5 + 8) & 1;
          if ( v95 )
          {
            v76 = a5 + 16;
            v77 = 15;
          }
          else
          {
            v96 = *(unsigned int *)(a5 + 24);
            v76 = *(_QWORD *)(a5 + 16);
            if ( !(_DWORD)v96 )
              goto LABEL_84;
            v77 = v96 - 1;
          }
          v78 = v77 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
          v79 = v76 + 16LL * v78;
          v80 = *(_QWORD *)v79;
          if ( v94 != *(_QWORD *)v79 )
          {
            v111 = 1;
            while ( v80 != -4096 )
            {
              v128 = v111 + 1;
              v78 = v77 & (v111 + v78);
              v79 = v76 + 16LL * v78;
              v80 = *(_QWORD *)v79;
              if ( v94 == *(_QWORD *)v79 )
                goto LABEL_50;
              v111 = v128;
            }
            if ( v95 )
            {
              v105 = 256;
            }
            else
            {
              v96 = *(unsigned int *)(a5 + 24);
LABEL_84:
              v105 = 16 * v96;
            }
            v79 = v76 + v105;
          }
LABEL_50:
          v81 = 256;
          if ( !v95 )
            v81 = 16LL * *(unsigned int *)(a5 + 24);
          v82 = 0.0;
          if ( v79 != v76 + v81 )
            v82 = (double)*(int *)(*(_QWORD *)(a5 + 272) + 16LL * *(unsigned int *)(v79 + 8) + 8);
          v83 = sub_AD8DD0(*(_QWORD *)(v89 + 8), v82);
          v85 = (unsigned int)v147;
          v86 = (unsigned int)v147 + 1LL;
          if ( v86 > HIDWORD(v147) )
          {
            v133 = v83;
            sub_C8D5F0((__int64)&v146, v148, (unsigned int)v147 + 1LL, 8u, v86, v84);
            v85 = (unsigned int)v147;
            v83 = v133;
          }
          ++v13;
          v146[v85] = (__int64)v83;
          v87 = (unsigned int)(v147 + 1);
          LODWORD(v147) = v147 + 1;
          if ( v75 == v13 )
          {
            v106 = v146;
            goto LABEL_87;
          }
        }
      }
      v106 = (__int64 *)v148;
      v87 = 0;
LABEL_87:
      v107 = sub_AD3730(v106, v87);
      v108 = v107;
      v143 = 257;
      if ( *(_BYTE *)(a3 + 108) )
      {
        v8 = sub_B35400(a3, 0x6Cu, v8, v107, v141, (__int64)v142, 0, 0, 0);
      }
      else
      {
        v109 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a3 + 80) + 40LL))(
                 *(_QWORD *)(a3 + 80),
                 18,
                 v8,
                 v107,
                 *(unsigned int *)(a3 + 104));
        if ( v109 )
        {
          v8 = v109;
        }
        else
        {
          v145 = 257;
          v113 = *(_DWORD *)(a3 + 104);
          v114 = sub_B504D0(18, v8, v108, (__int64)v144, 0, 0);
          v115 = *(_QWORD *)(a3 + 96);
          v8 = v114;
          if ( v115 )
            sub_B99FD0(v114, 3u, v115);
          sub_B45150(v8, v113);
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
            *(_QWORD *)(a3 + 88),
            v8,
            v142,
            *(_QWORD *)(a3 + 56),
            *(_QWORD *)(a3 + 64));
          v116 = *(unsigned int **)a3;
          v117 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
          if ( *(_QWORD *)a3 != v117 )
          {
            do
            {
              v118 = *((_QWORD *)v116 + 1);
              v119 = *v116;
              v116 += 4;
              sub_B99FD0(v8, v119, v118);
            }
            while ( (unsigned int *)v117 != v116 );
          }
        }
      }
      if ( v146 != (__int64 *)v148 )
        _libc_free((unsigned __int64)v146);
      return v8;
    default:
      return 0;
  }
}
