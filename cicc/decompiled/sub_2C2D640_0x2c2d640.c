// Function: sub_2C2D640
// Address: 0x2c2d640
//
unsigned __int64 *__fastcall sub_2C2D640(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r15
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r14
  _BYTE *v28; // rsi
  __int64 v29; // rdx
  _QWORD *v30; // rdi
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r15
  _BYTE *v37; // rsi
  __int64 v38; // rdx
  _QWORD *v39; // rdi
  __int64 v40; // r11
  _QWORD *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r10
  __int64 v47; // r11
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r9
  __int64 v55; // r8
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // r13
  __int64 v70; // r12
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rax
  _BYTE *v74; // rsi
  int v75; // ecx
  __int64 v76; // rdi
  int v77; // ecx
  unsigned int v78; // edx
  _QWORD *v79; // rax
  _BYTE *v80; // r10
  __int64 v81; // rdi
  __int64 v82; // r14
  __int64 v83; // rax
  __int64 v84; // rdi
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 v87; // rax
  _BYTE *v88; // rsi
  int v89; // ecx
  __int64 v90; // rdi
  int v91; // ecx
  unsigned int v92; // edx
  _QWORD *v93; // rax
  _BYTE *v94; // r8
  __int64 v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // rsi
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  unsigned __int64 v105; // rax
  __int64 *v106; // r14
  __int64 *v107; // rax
  __int64 v108; // r8
  __int64 v109; // rcx
  _QWORD *v110; // r13
  __int64 v111; // r13
  _QWORD *v112; // r13
  __int64 v113; // rbx
  int v115; // eax
  int v116; // r8d
  int v117; // eax
  int v118; // r9d
  __int64 v120; // [rsp+18h] [rbp-128h]
  __int64 v121; // [rsp+20h] [rbp-120h]
  __int64 v122; // [rsp+28h] [rbp-118h]
  __int64 v123; // [rsp+30h] [rbp-110h]
  __int64 v124; // [rsp+30h] [rbp-110h]
  __int64 v125; // [rsp+40h] [rbp-100h]
  __int64 v126; // [rsp+48h] [rbp-F8h]
  __int64 v127; // [rsp+50h] [rbp-F0h]
  _QWORD *v128; // [rsp+58h] [rbp-E8h]
  __int64 v130; // [rsp+60h] [rbp-E0h]
  __int64 v132; // [rsp+78h] [rbp-C8h]
  _QWORD *v133; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v134; // [rsp+88h] [rbp-B8h] BYREF
  __int64 v135; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 *v136; // [rsp+98h] [rbp-A8h]
  __int64 v137[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v138[2]; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE *v139; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v140; // [rsp+C8h] [rbp-78h]
  _QWORD v141[2]; // [rsp+D0h] [rbp-70h] BYREF
  void *v142[4]; // [rsp+E0h] [rbp-60h] BYREF
  __int16 v143; // [rsp+100h] [rbp-40h]

  v127 = sub_2BF3F10(a1);
  v122 = *(_QWORD *)(v127 + 120);
  v8 = sub_2BF0A50(v122);
  v9 = *(_QWORD *)(v8 + 80);
  v136 = (unsigned __int64 *)(v8 + 24);
  v10 = a1[1];
  v135 = v9;
  if ( *(_DWORD *)(v10 + 64) != 1 )
    BUG();
  v128 = **(_QWORD ***)(**(_QWORD **)(v10 + 56) + 56LL);
  v11 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == a4 + 48 )
    goto LABEL_96;
  if ( !v11 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_96:
    BUG();
  v12 = *(_QWORD *)(v11 - 56);
  v13 = *(_QWORD *)(v11 - 88);
  if ( *(_BYTE *)(a3 + 84) )
  {
    v14 = *(_QWORD **)(a3 + 64);
    v15 = &v14[*(unsigned int *)(a3 + 76)];
    if ( v14 == v15 )
    {
LABEL_74:
      v16 = v12;
    }
    else
    {
      while ( v12 != *v14 )
      {
        if ( v15 == ++v14 )
          goto LABEL_74;
      }
      v16 = v13;
    }
  }
  else
  {
    v16 = *(_QWORD *)(v11 - 88);
    if ( !sub_C8CA60(a3 + 56, *(_QWORD *)(v11 - 56)) )
      v16 = v12;
  }
  v123 = sub_2BF0B50((__int64)a1, v16);
  if ( *(_BYTE *)(a3 + 84) )
  {
    v17 = *(_QWORD **)(a3 + 64);
    v18 = &v17[*(unsigned int *)(a3 + 76)];
    if ( v17 != v18 )
    {
      while ( v12 != *v17 )
      {
        if ( v18 == ++v17 )
          goto LABEL_17;
      }
      v13 = v12;
    }
  }
  else if ( sub_C8CA60(a3 + 56, v12) )
  {
    v13 = v12;
  }
LABEL_17:
  v137[0] = sub_2AB6F10(a5, v13);
  v143 = 257;
  v134 = 0;
  v138[0] = 0;
  v139 = 0;
  v19 = sub_2AAFFE0(&v135, 70, v137, 1, (__int64 *)&v139, v142);
  sub_9C6650(&v139);
  v20 = v19 + 96;
  if ( !v19 )
    v20 = 0;
  v125 = v20;
  sub_9C6650(v138);
  sub_9C6650(&v134);
  v143 = 257;
  v138[0] = 0;
  v137[0] = v125;
  v21 = sub_2C27AE0(&v135, 85, v137, 1, (int)v139, 0, v138, v142);
  v22 = v21 + 12;
  if ( !v21 )
    v22 = 0;
  v126 = (__int64)v22;
  sub_9C6650(v138);
  v142[0] = "middle.split";
  v143 = 259;
  v27 = sub_22077B0(0x80u);
  if ( v27 )
  {
    sub_CA0F50((__int64 *)&v139, v142);
    v28 = v139;
    *(_BYTE *)(v27 + 8) = 1;
    v29 = v140;
    *(_QWORD *)v27 = &unk_4A23970;
    *(_QWORD *)(v27 + 16) = v27 + 32;
    sub_2C256A0((__int64 *)(v27 + 16), v28, (__int64)&v28[v29]);
    v30 = v139;
    v23 = v27 + 96;
    *(_QWORD *)(v27 + 56) = v27 + 72;
    *(_QWORD *)(v27 + 64) = 0x100000000LL;
    *(_QWORD *)(v27 + 88) = 0x100000000LL;
    *(_QWORD *)(v27 + 48) = 0;
    *(_QWORD *)(v27 + 80) = v27 + 96;
    *(_QWORD *)(v27 + 104) = 0;
    if ( v30 != v141 )
      j_j___libc_free_0((unsigned __int64)v30);
    v24 = v27 + 112;
    v121 = v27 + 112;
    *(_QWORD *)(v27 + 120) = v27 + 112;
    *(_QWORD *)v27 = &unk_4A23A00;
    *(_QWORD *)(v27 + 112) = (v27 + 112) | 4;
  }
  else
  {
    v121 = 112;
  }
  v31 = (__int64)(a1 + 74);
  sub_2AB9570(v31, v27, v23, v24, v25, v26);
  v142[0] = "vector.early.exit";
  v143 = 259;
  v36 = sub_22077B0(0x80u);
  if ( v36 )
  {
    sub_CA0F50((__int64 *)&v139, v142);
    v37 = v139;
    *(_BYTE *)(v36 + 8) = 1;
    v38 = v140;
    *(_QWORD *)v36 = &unk_4A23970;
    *(_QWORD *)(v36 + 16) = v36 + 32;
    sub_2C256A0((__int64 *)(v36 + 16), v37, (__int64)&v37[v38]);
    v39 = v139;
    v32 = v36 + 96;
    *(_QWORD *)(v36 + 56) = v36 + 72;
    *(_QWORD *)(v36 + 64) = 0x100000000LL;
    *(_QWORD *)(v36 + 88) = 0x100000000LL;
    *(_QWORD *)(v36 + 48) = 0;
    *(_QWORD *)(v36 + 80) = v36 + 96;
    *(_QWORD *)(v36 + 104) = 0;
    if ( v39 != v141 )
      j_j___libc_free_0((unsigned __int64)v39);
    v33 = v36 + 112;
    v120 = v36 + 112;
    *(_QWORD *)(v36 + 120) = v36 + 112;
    *(_QWORD *)v36 = &unk_4A23A00;
    *(_QWORD *)(v36 + 112) = (v36 + 112) | 4;
  }
  else
  {
    v120 = 112;
  }
  sub_2AB9570(v31, v36, v32, v33, v34, v35);
  v142[0] = v128;
  v139 = (_BYTE *)v127;
  sub_2C25750(*(_QWORD **)(v127 + 80), *(_QWORD *)(v127 + 80) + 8LL * *(unsigned int *)(v127 + 88), (__int64 *)v142);
  v41 = sub_2C25750(*(_QWORD **)(v40 + 56), *(_QWORD *)(v40 + 56) + 8LL * *(unsigned int *)(v40 + 64), (__int64 *)&v139);
  v48 = ((__int64)v41 - v47) >> 3;
  if ( (_DWORD)v45 == -1 )
  {
    sub_2AB9570(v127 + 80, v27, v42, v43, v44, v45);
  }
  else
  {
    v45 = (unsigned int)v45;
    *(_QWORD *)(v46 + 8LL * (unsigned int)v45) = v27;
  }
  sub_2AB9570(v27 + 56, v127, v42, v43, v44, v45);
  sub_2AB9570(v27 + 80, (__int64)v128, v49, v50, v27 + 80, v51);
  v55 = v27 + 80;
  if ( (_DWORD)v48 == -1 )
  {
    sub_2AB9570((__int64)(v128 + 7), v27, v52, v53, v55, v54);
    v55 = v27 + 80;
  }
  else
  {
    *(_QWORD *)(v128[7] + 8LL * (unsigned int)v48) = v27;
  }
  sub_2AB9570(v55, v36, v52, v53, v55, v54);
  sub_2AB9570(v36 + 56, v27, v56, v57, v58, v59);
  v60 = *(__int64 **)(v27 + 80);
  v61 = *v60;
  *v60 = v60[1];
  v60[1] = v61;
  sub_2AB9570(v36 + 80, v123, v61, v62, v63, v64);
  sub_2AB9570(v123 + 56, v36, v65, v66, v67, v68);
  v137[0] = v27;
  v138[0] = v36;
  v137[1] = v121;
  v138[1] = v120;
  v69 = *(_QWORD *)(v123 + 120);
  v124 = v123 + 112;
  while ( v124 != v69 )
  {
    if ( !v69 )
      BUG();
    v70 = *(_QWORD *)(v69 + 72);
    if ( *(_BYTE *)v70 != 84 )
      break;
    v71 = *(_QWORD *)(v70 - 8);
    v72 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v70 + 4) & 0x7FFFFFF) != 0 )
    {
      v73 = 0;
      do
      {
        if ( a4 == *(_QWORD *)(v71 + 32LL * *(unsigned int *)(v70 + 72) + 8 * v73) )
        {
          v72 = 32 * v73;
          goto LABEL_41;
        }
        ++v73;
      }
      while ( (*(_DWORD *)(v70 + 4) & 0x7FFFFFF) != (_DWORD)v73 );
      v72 = 0x1FFFFFFFE0LL;
    }
LABEL_41:
    v74 = *(_BYTE **)(v71 + v72);
    if ( *v74 > 0x1Cu )
    {
      v75 = *(_DWORD *)(a5 + 152);
      v76 = *(_QWORD *)(a5 + 136);
      if ( v75 )
      {
        v77 = v75 - 1;
        v78 = v77 & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
        v79 = (_QWORD *)(v76 + 16LL * v78);
        v80 = (_BYTE *)*v79;
        if ( v74 == (_BYTE *)*v79 )
        {
LABEL_44:
          v81 = v79[1];
          if ( v81 )
          {
            v82 = (__int64)sub_2C2A370((_QWORD *)(v81 + 16), 0);
            goto LABEL_46;
          }
        }
        else
        {
          v115 = 1;
          while ( v80 != (_BYTE *)-4096LL )
          {
            v116 = v115 + 1;
            v78 = v77 & (v115 + v78);
            v79 = (_QWORD *)(v76 + 16LL * v78);
            v80 = (_BYTE *)*v79;
            if ( v74 == (_BYTE *)*v79 )
              goto LABEL_44;
            v115 = v116;
          }
        }
      }
    }
    v82 = sub_2AC42A0(*(_QWORD *)a5, (__int64)v74);
LABEL_46:
    if ( sub_D47600(a3) )
    {
      v83 = sub_D47930(a3);
      v84 = *(_QWORD *)(v70 - 8);
      v85 = v83;
      if ( (*(_DWORD *)(v70 + 4) & 0x7FFFFFF) != 0 )
      {
        v86 = 0;
        while ( v85 != *(_QWORD *)(v84 + 32LL * *(unsigned int *)(v70 + 72) + 8 * v86) )
        {
          if ( (*(_DWORD *)(v70 + 4) & 0x7FFFFFF) == (_DWORD)++v86 )
            goto LABEL_82;
        }
        v87 = 32 * v86;
      }
      else
      {
LABEL_82:
        v87 = 0x1FFFFFFFE0LL;
      }
      v88 = *(_BYTE **)(v84 + v87);
      if ( *v88 > 0x1Cu )
      {
        v89 = *(_DWORD *)(a5 + 152);
        v90 = *(_QWORD *)(a5 + 136);
        if ( v89 )
        {
          v91 = v89 - 1;
          v92 = v91 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
          v93 = (_QWORD *)(v90 + 16LL * v92);
          v94 = (_BYTE *)*v93;
          if ( (_BYTE *)*v93 == v88 )
          {
LABEL_55:
            v95 = v93[1];
            if ( v95 )
            {
              v98 = (__int64)sub_2C2A370((_QWORD *)(v95 + 16), 0);
LABEL_57:
              sub_2AAECA0(v69 + 16, v98, v96, v97, v99, v100);
              sub_2C1BE50(v69 - 24, v137);
              goto LABEL_58;
            }
          }
          else
          {
            v117 = 1;
            while ( v94 != (_BYTE *)-4096LL )
            {
              v118 = v117 + 1;
              v92 = v91 & (v117 + v92);
              v93 = (_QWORD *)(v90 + 16LL * v92);
              v94 = (_BYTE *)*v93;
              if ( v88 == (_BYTE *)*v93 )
                goto LABEL_55;
              v117 = v118;
            }
          }
        }
      }
      v98 = sub_2AC42A0(*(_QWORD *)a5, (__int64)v88);
      goto LABEL_57;
    }
LABEL_58:
    if ( sub_2BF04A0(v82) )
    {
      v139 = (_BYTE *)v82;
      v143 = 257;
      v133 = 0;
      v140 = v125;
      v82 = (__int64)sub_2C27AE0(v138, 86, (__int64 *)&v139, 2, v134, 0, (__int64 *)&v133, v142);
      if ( v82 )
        v82 += 96;
      sub_9C6650(&v133);
    }
    sub_2AAECA0(v69 + 16, v82, v101, v102, v103, v104);
    v69 = *(_QWORD *)(v69 + 8);
  }
  v143 = 257;
  v134 = 0;
  v133 = (_QWORD *)v126;
  sub_2C27AE0(v137, 79, (__int64 *)&v133, 1, (int)v139, 0, &v134, v142);
  sub_9C6650(&v134);
  v105 = sub_2BF0A50(v122);
  v134 = 0;
  v106 = (__int64 *)v105;
  v143 = 257;
  v107 = *(__int64 **)(v105 + 48);
  v108 = v107[1];
  v109 = *v107;
  v139 = 0;
  v130 = v108;
  v132 = v109;
  v110 = (_QWORD *)sub_22077B0(0xC8u);
  if ( v110 )
  {
    sub_2C1A5F0((__int64)v110, 53, 32, v132, v130, (__int64 *)&v139, v142);
    if ( v135 )
      sub_2AAFF40(v135, v110, v136);
    v111 = (__int64)(v110 + 12);
  }
  else
  {
    v111 = v135;
    if ( v135 )
    {
      v111 = 0;
      sub_2AAFF40(v135, 0, v136);
    }
  }
  sub_9C6650(&v139);
  sub_9C6650(&v134);
  v143 = 257;
  v140 = v111;
  v133 = 0;
  v139 = (_BYTE *)v126;
  v112 = sub_2C27AE0(&v135, 29, (__int64 *)&v139, 2, v134, 0, (__int64 *)&v133, v142);
  sub_9C6650(&v133);
  if ( v112 )
    v112 += 12;
  v134 = 0;
  v143 = 257;
  v133 = v112;
  v139 = 0;
  v113 = sub_2AAFFE0(&v135, 79, (__int64 *)&v133, 1, (__int64 *)&v139, v142);
  sub_9C6650(&v139);
  *(_QWORD *)(v113 + 136) = 0;
  sub_9C6650(&v134);
  return sub_2C19E60(v106);
}
