// Function: sub_FF6E20
// Address: 0xff6e20
//
__int64 __fastcall sub_FF6E20(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned __int64 v5; // rax
  unsigned int v6; // r15d
  unsigned int v7; // ebx
  bool v8; // al
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // cl
  char *v12; // rdx
  char *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned int v19; // ebx
  __int64 v20; // r15
  _QWORD *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rcx
  int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r13
  unsigned int v27; // r12d
  _QWORD *v28; // rdi
  unsigned int v29; // r12d
  __int64 *v31; // rax
  unsigned __int8 *v32; // rbx
  char v33; // al
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  _QWORD *v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rbx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 *v43; // rdx
  __int64 v44; // r12
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int8 v50; // al
  _DWORD *v51; // rax
  _DWORD *v52; // rdx
  unsigned __int64 v53; // rax
  __int64 v54; // r14
  int v55; // r13d
  unsigned int v56; // ebx
  unsigned int v57; // r15d
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // r13
  char *v61; // rbx
  unsigned __int8 *v62; // rcx
  __int64 v63; // rax
  unsigned __int8 *v64; // rax
  unsigned __int8 **v65; // rax
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  _BYTE *v70; // r13
  __int64 v71; // rdx
  char *v72; // rax
  int v73; // eax
  char v74; // dl
  int v75; // eax
  __int64 v76; // r12
  unsigned __int8 *v77; // rdx
  _QWORD *v78; // rax
  _QWORD *v79; // rdx
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  unsigned __int8 *v82; // rbx
  __int64 v84; // [rsp+8h] [rbp-218h]
  __int64 v85; // [rsp+20h] [rbp-200h]
  __int64 v86; // [rsp+28h] [rbp-1F8h]
  int v87; // [rsp+30h] [rbp-1F0h]
  __int64 *v88; // [rsp+38h] [rbp-1E8h]
  __int64 v89; // [rsp+40h] [rbp-1E0h]
  unsigned int v90; // [rsp+54h] [rbp-1CCh]
  char v91; // [rsp+58h] [rbp-1C8h]
  int v92; // [rsp+58h] [rbp-1C8h]
  unsigned __int8 *v93; // [rsp+58h] [rbp-1C8h]
  __int64 v94; // [rsp+60h] [rbp-1C0h]
  __int64 v95; // [rsp+60h] [rbp-1C0h]
  int v96; // [rsp+68h] [rbp-1B8h]
  __int64 *v97; // [rsp+68h] [rbp-1B8h]
  unsigned __int64 v98; // [rsp+70h] [rbp-1B0h]
  unsigned __int64 v99; // [rsp+70h] [rbp-1B0h]
  unsigned __int64 v100; // [rsp+80h] [rbp-1A0h]
  __int64 v101; // [rsp+80h] [rbp-1A0h]
  int v102; // [rsp+80h] [rbp-1A0h]
  unsigned __int64 v103; // [rsp+88h] [rbp-198h]
  char v104; // [rsp+88h] [rbp-198h]
  __int64 v105; // [rsp+98h] [rbp-188h]
  char v106[8]; // [rsp+A0h] [rbp-180h] BYREF
  unsigned __int64 v107; // [rsp+A8h] [rbp-178h]
  char *v108; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-158h]
  _BYTE v110[16]; // [rsp+D0h] [rbp-150h] BYREF
  _QWORD *v111; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v112; // [rsp+E8h] [rbp-138h]
  _QWORD v113[8]; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v114; // [rsp+130h] [rbp-F0h] BYREF
  char *v115; // [rsp+138h] [rbp-E8h]
  __int64 v116; // [rsp+140h] [rbp-E0h]
  int v117; // [rsp+148h] [rbp-D8h]
  char v118; // [rsp+14Ch] [rbp-D4h]
  char v119; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v120; // [rsp+190h] [rbp-90h] BYREF
  __int64 v121; // [rsp+198h] [rbp-88h]
  __int64 v122; // [rsp+1A0h] [rbp-80h] BYREF
  int v123; // [rsp+1A8h] [rbp-78h]
  char v124; // [rsp+1ACh] [rbp-74h]
  unsigned __int8 *v125; // [rsp+1B0h] [rbp-70h] BYREF

  v2 = a1;
  v89 = a2;
  sub_FEF2D0((__int64)v106, a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
  v114 = 0;
  v115 = &v119;
  v116 = 8;
  v90 = qword_4F8E7C8;
  v5 = a2;
  v117 = 0;
  v118 = 1;
  v98 = v107;
  v100 = a2 + 48;
  if ( !v107 )
    goto LABEL_69;
  a2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v103 = a2;
  if ( a2 == v100 || !a2 || (unsigned int)*(unsigned __int8 *)(a2 - 24) - 30 > 0xA )
    BUG();
  if ( *(_BYTE *)(a2 - 24) != 31 )
    goto LABEL_7;
  if ( (*(_DWORD *)(a2 - 20) & 0x7FFFFFF) != 3 )
    goto LABEL_7;
  v86 = *(_QWORD *)(a2 - 120);
  if ( (unsigned __int8)(*(_BYTE *)v86 - 82) > 1u )
    goto LABEL_7;
  v32 = *(unsigned __int8 **)(v86 - 64);
  v33 = *v32;
  if ( *v32 <= 0x1Cu )
    goto LABEL_7;
  a2 = *(_QWORD *)(v86 - 32);
  v84 = a2;
  if ( *(_BYTE *)a2 > 0x15u )
    goto LABEL_7;
  v108 = v110;
  v109 = 0x100000000LL;
  if ( v33 == 84 )
  {
    v85 = v107 + 56;
  }
  else
  {
    v75 = *v32;
    v76 = v107 + 56;
    do
    {
      if ( (unsigned int)(v75 - 42) <= 0x11 )
      {
        v77 = (v32[7] & 0x40) != 0
            ? (unsigned __int8 *)*((_QWORD *)v32 - 1)
            : &v32[-32 * (*((_DWORD *)v32 + 1) & 0x7FFFFFF)];
        if ( **((_BYTE **)v77 + 4) <= 0x15u )
        {
          a2 = *((_QWORD *)v32 + 5);
          if ( *(_BYTE *)(v98 + 84) )
          {
            v78 = *(_QWORD **)(v98 + 64);
            v79 = &v78[*(unsigned int *)(v98 + 76)];
            if ( v78 == v79 )
              goto LABEL_66;
            while ( a2 != *v78 )
            {
              if ( v79 == ++v78 )
                goto LABEL_66;
            }
          }
          else if ( !sub_C8CA60(v76, a2) )
          {
            goto LABEL_66;
          }
          v80 = (unsigned int)v109;
          v81 = (unsigned int)v109 + 1LL;
          if ( v81 > HIDWORD(v109) )
          {
            a2 = (unsigned __int64)v110;
            sub_C8D5F0((__int64)&v108, v110, v81, 8u, v3, v4);
            v80 = (unsigned int)v109;
          }
          *(_QWORD *)&v108[8 * v80] = v32;
          LODWORD(v109) = v109 + 1;
          v82 = (v32[7] & 0x40) != 0
              ? (unsigned __int8 *)*((_QWORD *)v32 - 1)
              : &v32[-32 * (*((_DWORD *)v32 + 1) & 0x7FFFFFF)];
          v32 = *(unsigned __int8 **)v82;
          v75 = *v32;
          if ( (unsigned __int8)v75 > 0x1Cu )
            continue;
        }
      }
      goto LABEL_66;
    }
    while ( (_BYTE)v75 != 84 );
    v85 = v76;
  }
  a2 = *((_QWORD *)v32 + 5);
  if ( *(_BYTE *)(v98 + 84) )
  {
    v34 = *(_QWORD **)(v98 + 64);
    v35 = &v34[*(unsigned int *)(v98 + 76)];
    if ( v34 == v35 )
      goto LABEL_66;
    while ( a2 != *v34 )
    {
      if ( v35 == ++v34 )
        goto LABEL_66;
    }
    goto LABEL_84;
  }
  if ( sub_C8CA60(v85, a2) )
  {
LABEL_84:
    v124 = 1;
    v112 = 0x800000001LL;
    v113[0] = v32;
    v122 = 0x100000008LL;
    v123 = 0;
    v125 = v32;
    v120 = 1;
    v121 = (__int64)&v125;
    v36 = v113;
    v111 = v113;
    v37 = 1;
    while ( 1 )
    {
      v38 = v37--;
      v39 = v36[v38 - 1];
      LODWORD(v112) = v37;
      v95 = v39;
      v40 = *(_QWORD *)(v39 - 8);
      v41 = 32LL * *(unsigned int *)(v39 + 72);
      v42 = v41 + 8LL * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF);
      v43 = (__int64 *)(v40 + v41);
      a2 = v40 + v42;
      v88 = (__int64 *)(v40 + v42);
      if ( (__int64 *)(v40 + v42) == v43 )
        goto LABEL_101;
      v97 = v43;
      do
      {
        v44 = *v97;
        if ( *(_BYTE *)(v98 + 84) )
        {
          v45 = *(_QWORD **)(v98 + 64);
          v46 = &v45[*(unsigned int *)(v98 + 76)];
          if ( v45 == v46 )
            goto LABEL_99;
          while ( v44 != *v45 )
          {
            if ( v46 == ++v45 )
              goto LABEL_99;
          }
        }
        else
        {
          a2 = *v97;
          if ( !sub_C8CA60(v85, *v97) )
            goto LABEL_99;
        }
        v92 = *(_DWORD *)(v95 + 4);
        a2 = *(_QWORD *)(v95 - 8);
        v47 = 0x1FFFFFFFE0LL;
        v48 = v92 & 0x7FFFFFF;
        if ( (v92 & 0x7FFFFFF) != 0 )
        {
          v49 = 0;
          v42 = a2 + 32LL * *(unsigned int *)(v95 + 72);
          do
          {
            if ( v44 == *(_QWORD *)(v42 + 8 * v49) )
            {
              v47 = 32 * v49;
              goto LABEL_97;
            }
            ++v49;
          }
          while ( (_DWORD)v48 != (_DWORD)v49 );
          v47 = 0x1FFFFFFFE0LL;
        }
LABEL_97:
        v93 = *(unsigned __int8 **)(a2 + v47);
        v50 = *v93;
        if ( *v93 <= 0x1Cu )
        {
          if ( v50 > 0x15u )
            goto LABEL_99;
          v53 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v100 == v53 )
            goto LABEL_99;
          if ( !v53 )
            goto LABEL_109;
          v54 = v53 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v53 - 24) - 30 > 0xA )
            goto LABEL_99;
          v87 = sub_B46E30(v54);
          v55 = v87 >> 2;
          if ( v87 >> 2 > 0 )
          {
            v56 = 0;
            while ( 1 )
            {
              a2 = v56;
              if ( v44 == sub_B46EC0(v54, v56) )
                goto LABEL_127;
              v57 = v56 + 1;
              a2 = v56 + 1;
              if ( v44 == sub_B46EC0(v54, a2) )
                goto LABEL_128;
              v57 = v56 + 2;
              a2 = v56 + 2;
              if ( v44 == sub_B46EC0(v54, a2) )
                goto LABEL_128;
              v57 = v56 + 3;
              a2 = v56 + 3;
              if ( v44 == sub_B46EC0(v54, a2) )
                goto LABEL_128;
              v56 += 4;
              if ( !--v55 )
              {
                v73 = v87 - v56;
                goto LABEL_164;
              }
            }
          }
          v73 = v87;
          v56 = 0;
LABEL_164:
          switch ( v73 )
          {
            case 2:
LABEL_168:
              a2 = v56;
              if ( v44 == sub_B46EC0(v54, v56) )
                goto LABEL_127;
              ++v56;
              break;
            case 3:
              a2 = v56;
              if ( v44 == sub_B46EC0(v54, v56) )
                goto LABEL_127;
              ++v56;
              goto LABEL_168;
            case 1:
              break;
            default:
              goto LABEL_99;
          }
          a2 = v56;
          if ( v44 != sub_B46EC0(v54, v56) )
            goto LABEL_99;
LABEL_127:
          v57 = v56;
LABEL_128:
          if ( v87 != v57 )
          {
            v58 = sub_AA4E30(v89);
            v59 = (__int64)v108;
            v60 = v58;
            v61 = &v108[8 * (unsigned int)v109];
            if ( v108 == v61 )
            {
LABEL_152:
              a2 = (unsigned __int64)v93;
              v68 = sub_9719A0(*(_WORD *)(v86 + 2) & 0x3F, v93, v84, v60, 0, 0);
              v70 = (_BYTE *)v68;
              if ( v68
                && (sub_AD7890(v68, (__int64)v93, v69, v42, v3) && v44 == *(_QWORD *)(v103 - 56)
                 || sub_AD7A80(v70, (__int64)v93, v71, v42, v3) && v44 == *(_QWORD *)(v103 - 88)) )
              {
                if ( !v118 )
                  goto LABEL_197;
                v72 = v115;
                v71 = HIDWORD(v116);
                v42 = (__int64)&v115[8 * HIDWORD(v116)];
                if ( v115 == (char *)v42 )
                {
LABEL_161:
                  if ( HIDWORD(v116) < (unsigned int)v116 )
                  {
                    ++HIDWORD(v116);
                    *(_QWORD *)v42 = v44;
                    ++v114;
                    goto LABEL_99;
                  }
LABEL_197:
                  a2 = v44;
                  sub_C8CC70((__int64)&v114, v44, v71, v42, v3, v4);
                  goto LABEL_99;
                }
                while ( v44 != *(_QWORD *)v72 )
                {
                  v72 += 8;
                  if ( (char *)v42 == v72 )
                    goto LABEL_161;
                }
              }
            }
            else
            {
              a2 = (unsigned __int64)v93;
              while ( 1 )
              {
                v64 = (unsigned __int8 *)*((_QWORD *)v61 - 1);
                v62 = (v64[7] & 0x40) != 0
                    ? (unsigned __int8 *)*((_QWORD *)v64 - 1)
                    : &v64[-32 * (*((_DWORD *)v64 + 1) & 0x7FFFFFF)];
                v63 = sub_96E6C0((unsigned int)*v64 - 29, a2, *((_BYTE **)v62 + 4), v60);
                a2 = v63;
                if ( !v63 )
                  break;
                v61 -= 8;
                if ( (char *)v59 == v61 )
                {
                  v93 = (unsigned __int8 *)v63;
                  goto LABEL_152;
                }
              }
            }
          }
          goto LABEL_99;
        }
        if ( v50 != 84 )
          goto LABEL_99;
        if ( !v124 )
          goto LABEL_172;
        v65 = (unsigned __int8 **)v121;
        v42 = HIDWORD(v122);
        v48 = v121 + 8LL * HIDWORD(v122);
        if ( v121 == v48 )
        {
LABEL_144:
          if ( HIDWORD(v122) >= (unsigned int)v122 )
          {
LABEL_172:
            a2 = (unsigned __int64)v93;
            sub_C8CC70((__int64)&v120, (__int64)v93, v48, v42, v3, v4);
            if ( !v74 )
              goto LABEL_99;
          }
          else
          {
            ++HIDWORD(v122);
            *(_QWORD *)v48 = v93;
            ++v120;
          }
          v66 = (unsigned int)v112;
          v42 = HIDWORD(v112);
          v67 = (unsigned int)v112 + 1LL;
          if ( v67 > HIDWORD(v112) )
          {
            a2 = (unsigned __int64)v113;
            sub_C8D5F0((__int64)&v111, v113, v67, 8u, v3, v4);
            v66 = (unsigned int)v112;
          }
          v111[v66] = v93;
          LODWORD(v112) = v112 + 1;
          goto LABEL_99;
        }
        a2 = (unsigned __int64)v93;
        while ( v93 != *v65 )
        {
          if ( (unsigned __int8 **)v48 == ++v65 )
            goto LABEL_144;
        }
LABEL_99:
        ++v97;
      }
      while ( v88 != v97 );
      v37 = v112;
      v36 = v111;
LABEL_101:
      if ( !v37 )
      {
        v2 = a1;
        if ( v36 != v113 )
          _libc_free(v36, a2);
        if ( !v124 )
          _libc_free(v121, a2);
        if ( v108 != v110 )
          _libc_free(v108, a2);
        v103 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_7;
      }
    }
  }
LABEL_66:
  if ( v108 != v110 )
    _libc_free(v108, a2);
  v5 = v89;
LABEL_69:
  v103 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_7:
  v111 = v113;
  v112 = 0x400000000LL;
  if ( v100 == v103 )
    goto LABEL_115;
  if ( !v103 )
LABEL_109:
    BUG();
  v94 = v103 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v103 - 24) - 30 > 0xA || (v96 = sub_B46E30(v103 - 24)) == 0 )
  {
LABEL_115:
    v29 = 0;
    goto LABEL_49;
  }
  v6 = 0;
  v91 = 0;
  v99 = 0;
  do
  {
    v101 = sub_B46EC0(v94, v6);
    sub_FEF2D0((__int64)&v120, v101, *(_QWORD *)(v2 + 72), *(_QWORD *)(v2 + 80));
    v109 = (__int64)&v120;
    v108 = v106;
    a2 = (unsigned __int64)&v108;
    v105 = sub_FEF7A0(v2, (__int64 *)&v108);
    v7 = v105;
    v8 = sub_FEF3D0(v2, (__int64 *)&v108);
    v10 = v101;
    v11 = v8;
    if ( v8 )
    {
      a2 = v107;
      if ( !BYTE4(v105) )
      {
        v7 = 0xFFFFF;
LABEL_15:
        v9 = 1;
        v7 /= v90;
        if ( !v7 )
          v7 = 1;
        if ( !v107 )
          goto LABEL_56;
        goto LABEL_18;
      }
      if ( (_DWORD)v105 )
        goto LABEL_15;
      v11 = BYTE4(v105);
      if ( !v107 )
      {
        v91 = BYTE4(v105);
        goto LABEL_26;
      }
    }
    else
    {
      v11 = BYTE4(v105);
      if ( !v107 )
        goto LABEL_53;
    }
LABEL_18:
    if ( v118 )
    {
      v12 = v115;
      v13 = &v115[8 * HIDWORD(v116)];
      if ( v115 == v13 )
        goto LABEL_53;
      while ( v101 != *(_QWORD *)v12 )
      {
        v12 += 8;
        if ( v13 == v12 )
          goto LABEL_53;
      }
      if ( !v11 )
      {
LABEL_24:
        v14 = 0x7FFFF;
        v7 = 0x7FFFF;
LABEL_25:
        v99 += v14;
        v91 = 1;
        goto LABEL_26;
      }
    }
    else
    {
      a2 = v101;
      v104 = v11;
      v31 = sub_C8CA60((__int64)&v114, v101);
      v11 = v104;
      if ( !v31 )
      {
LABEL_53:
        if ( !v11 )
        {
          v99 += 0xFFFFFLL;
          v7 = 0xFFFFF;
          goto LABEL_26;
        }
LABEL_56:
        v91 = v11;
        v99 += v7;
        goto LABEL_26;
      }
      if ( !v104 )
        goto LABEL_24;
    }
    v91 = v11;
    if ( v7 )
    {
      v7 >>= 1;
      if ( !v7 )
        v7 = 1;
      v14 = v7;
      goto LABEL_25;
    }
LABEL_26:
    v15 = (unsigned int)v112;
    v16 = (unsigned int)v112 + 1LL;
    if ( v16 > HIDWORD(v112) )
    {
      a2 = (unsigned __int64)v113;
      sub_C8D5F0((__int64)&v111, v113, v16, 4u, v9, v10);
      v15 = (unsigned int)v112;
    }
    ++v6;
    *((_DWORD *)v111 + v15) = v7;
    v17 = (unsigned int)v112;
    v18 = (unsigned int)(v112 + 1);
    LODWORD(v112) = v112 + 1;
  }
  while ( v6 != v96 );
  v19 = v99;
  if ( v91 != 1 || !v99 )
  {
    v28 = v111;
    v29 = 0;
    goto LABEL_47;
  }
  v20 = (unsigned int)v18;
  if ( v99 <= 0xFFFFFFFF )
  {
    v120 = (__int64)&v122;
    v121 = 0x400000000LL;
    if ( (unsigned int)v18 > 4uLL )
      goto LABEL_110;
    if ( (_DWORD)v18 )
    {
      v10 = (__int64)v111;
      goto LABEL_38;
    }
    goto LABEL_72;
  }
  if ( !(_DWORD)v18 )
  {
    HIDWORD(v121) = 4;
    v120 = (__int64)&v122;
LABEL_72:
    LODWORD(v121) = v18;
    goto LABEL_44;
  }
  v21 = v111;
  v22 = 4 * v17 + 4;
  v19 = 0;
  v23 = 0;
  do
  {
    *(_DWORD *)((char *)v21 + v23) = *(unsigned int *)((char *)v21 + v23) / (v99 / 0xFFFFFFFF + 1);
    v21 = v111;
    v10 = (__int64)v111;
    v24 = *(_DWORD *)((char *)v111 + v23);
    if ( !v24 )
    {
      *(_DWORD *)((char *)v111 + v23) = 1;
      v21 = v111;
      v24 = *(_DWORD *)((char *)v111 + v23);
      v10 = (__int64)v111;
    }
    v23 += 4;
    v19 += v24;
  }
  while ( v23 != v22 );
  v120 = (__int64)&v122;
  v121 = 0x400000000LL;
  if ( (unsigned int)v18 <= 4uLL )
  {
LABEL_38:
    v25 = &v122;
    do
    {
      *(_DWORD *)v25 = -1;
      v25 = (__int64 *)((char *)v25 + 4);
      --v20;
    }
    while ( v20 );
    LODWORD(v121) = v18;
    goto LABEL_41;
  }
LABEL_110:
  v102 = v18;
  sub_C8D5F0((__int64)&v120, &v122, (unsigned int)v18, 4u, v18, v10);
  v51 = (_DWORD *)v120;
  LODWORD(v18) = v102;
  v52 = (_DWORD *)(v120 + 4 * v20);
  do
  {
    if ( v51 )
      *v51 = -1;
    ++v51;
  }
  while ( v51 != v52 );
  LODWORD(v121) = v102;
  v10 = (__int64)v111;
LABEL_41:
  v26 = 0;
  v27 = v18;
  while ( 1 )
  {
    sub_F02DB0(&v108, *(_DWORD *)(v10 + 4 * v26), v19);
    *(_DWORD *)(v120 + 4 * v26++) = (_DWORD)v108;
    if ( v27 <= (unsigned int)v26 )
      break;
    v10 = (__int64)v111;
  }
LABEL_44:
  a2 = v89;
  sub_FF6650(v2, v89, (__int64)&v120);
  if ( (__int64 *)v120 != &v122 )
    _libc_free(v120, v89);
  v28 = v111;
  v29 = 1;
LABEL_47:
  if ( v28 != v113 )
    _libc_free(v28, a2);
LABEL_49:
  if ( !v118 )
    _libc_free(v115, a2);
  return v29;
}
