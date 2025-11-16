// Function: sub_96F860
// Address: 0x96f860
//
__int64 __fastcall sub_96F860(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // r13
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdx
  _BYTE *v26; // rax
  unsigned int v27; // r12d
  _BYTE *v28; // r13
  unsigned __int8 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // r13
  __int64 v40; // rsi
  unsigned __int8 *v41; // r15
  __int64 v42; // rcx
  __int64 v43; // r13
  unsigned __int8 v44; // al
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // ebx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r13
  unsigned int v52; // r9d
  __int64 v53; // rsi
  unsigned __int64 v54; // rdx
  __int64 v55; // r8
  unsigned __int8 v56; // al
  __int64 v57; // rax
  __int64 v58; // rdx
  unsigned int v59; // ebx
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  char v68; // bl
  __int64 v69; // rdx
  int v70; // eax
  int v71; // ecx
  unsigned int v72; // ebx
  unsigned int v73; // r14d
  __int64 v74; // r13
  unsigned __int8 *v75; // rax
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rsi
  int v80; // edx
  _BYTE *v81; // rax
  _BYTE *v82; // rax
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // r14
  _QWORD *v87; // rbx
  __int64 v88; // rax
  __int64 v89; // rdx
  int v90; // eax
  int v91; // ecx
  unsigned __int8 *v92; // rax
  unsigned __int8 *v93; // rbx
  int v94; // edx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rbx
  int v98; // eax
  __int64 *v99; // rax
  __int64 *v100; // rdx
  unsigned int v101; // r13d
  int v102; // r14d
  __int64 v103; // rax
  __int64 v104; // rdx
  unsigned int v105; // eax
  __int64 v106; // [rsp+0h] [rbp-1D0h]
  int v107; // [rsp+Ch] [rbp-1C4h]
  __int64 v108; // [rsp+10h] [rbp-1C0h]
  int v109; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 *v110; // [rsp+28h] [rbp-1A8h]
  int v111; // [rsp+38h] [rbp-198h]
  __int64 v112; // [rsp+40h] [rbp-190h]
  __int64 v113; // [rsp+48h] [rbp-188h]
  int v114; // [rsp+48h] [rbp-188h]
  unsigned __int8 *v115; // [rsp+50h] [rbp-180h]
  int v116; // [rsp+50h] [rbp-180h]
  __int64 v117; // [rsp+50h] [rbp-180h]
  __int64 v118; // [rsp+58h] [rbp-178h]
  __int64 v119; // [rsp+58h] [rbp-178h]
  __int64 v120; // [rsp+58h] [rbp-178h]
  unsigned int v121; // [rsp+58h] [rbp-178h]
  __int64 v122; // [rsp+60h] [rbp-170h]
  unsigned int v123; // [rsp+60h] [rbp-170h]
  unsigned int v124; // [rsp+60h] [rbp-170h]
  int v125; // [rsp+60h] [rbp-170h]
  int v126; // [rsp+68h] [rbp-168h]
  unsigned int v127; // [rsp+6Ch] [rbp-164h]
  unsigned int v128; // [rsp+6Ch] [rbp-164h]
  unsigned int v129; // [rsp+6Ch] [rbp-164h]
  unsigned __int64 v130; // [rsp+70h] [rbp-160h] BYREF
  unsigned int v131; // [rsp+78h] [rbp-158h]
  __int64 v132; // [rsp+80h] [rbp-150h] BYREF
  __int64 v133; // [rsp+88h] [rbp-148h]
  __int64 v134; // [rsp+90h] [rbp-140h] BYREF
  __int64 v135; // [rsp+98h] [rbp-138h]
  _BYTE v136[304]; // [rsp+A0h] [rbp-130h] BYREF

  v6 = a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(unsigned __int8 *)(a2 + 8);
  v11 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v11 - 17) > 1 )
  {
    v12 = (unsigned __int8)v10;
    if ( (_BYTE)v10 == 17 )
    {
LABEL_35:
      if ( v11 != 18 && (unsigned __int8)(*(_BYTE *)a1 - 17) <= 1u )
      {
        v134 = a1;
        v41 = (unsigned __int8 *)sub_AD3730(&v134, 1);
        v38 = sub_96E500(v41, a2, (__int64)a3);
        if ( !v38 )
          return sub_96F860(v41, a2, a3);
        return v38;
      }
      goto LABEL_31;
    }
    goto LABEL_30;
  }
  if ( (unsigned __int8)v10 > 0xCu || (v42 = 4143, !_bittest64(&v42, v10)) )
  {
    v12 = (unsigned __int8)v10;
    if ( (v10 & 0xFD) != 4 )
    {
      if ( (unsigned __int8)v10 == 17 )
      {
LABEL_34:
        if ( v11 != 17 )
          goto LABEL_35;
        if ( (unsigned __int8)(*(_BYTE *)a1 - 16) <= 2u || *(_BYTE *)a1 == 11 )
        {
          v128 = *(_DWORD *)(a2 + 32);
          v124 = *(_DWORD *)(v9 + 32);
          if ( v128 != v124 )
          {
            v43 = *(_QWORD *)(a2 + 24);
            v44 = *(_BYTE *)(v43 + 8);
            if ( v44 <= 3u || v44 == 5 || (v44 & 0xFD) == 4 )
            {
              v45 = sub_BCAE30(*(_QWORD *)(a2 + 24));
              v135 = v46;
              v134 = v45;
              v47 = sub_CA1930(&v134);
              v49 = sub_BD5C60(a1, v9, v48);
              v50 = sub_BCCE00(v49, v47);
              v51 = sub_BCDA70(v50, v128);
              a1 = sub_96E500((unsigned __int8 *)a1, v51, (__int64)a3);
              if ( !a1 )
                a1 = sub_96F860(v6, v51, a3);
              v40 = a2;
              return sub_AD4C90(a1, v40, 0, v12, a5, a6);
            }
            v55 = *(_QWORD *)(v9 + 24);
            v56 = *(_BYTE *)(v55 + 8);
            if ( v56 <= 3u || v56 == 5 || (v56 & 0xFD) == 4 )
            {
              v118 = *(_QWORD *)(v9 + 24);
              v57 = sub_BCAE30(v118);
              v135 = v58;
              v134 = v57;
              v59 = sub_CA1930(&v134);
              v61 = sub_BD5C60(a1, v9, v60);
              v62 = sub_BCCE00(v61, v59);
              v63 = sub_BCDA70(v62, v124);
              v67 = sub_AD4C90(a1, v63, 0, v64, v65, v66);
              v55 = v118;
              v6 = v67;
            }
            v119 = v55;
            v68 = *a3;
            v134 = (__int64)v136;
            v135 = 0x2000000000LL;
            if ( v128 < v124 )
            {
              v112 = sub_AD6530(v43);
              v116 = v124 / v128;
              v132 = sub_BCAE30(v119);
              v133 = v69;
              v70 = sub_CA1930(&v132);
              v113 = a2;
              v109 = 0;
              v71 = 0;
              if ( v68 )
              {
                v71 = v70 * (v116 - 1);
                v70 = -v70;
              }
              v111 = v71;
              v72 = 0;
              v126 = v70;
              while ( 1 )
              {
                v73 = v111;
                v74 = v112;
                v125 = 0;
                do
                {
                  v75 = (unsigned __int8 *)sub_AD69F0(v6, v72);
                  v79 = (__int64)v75;
                  if ( !v75 )
                  {
LABEL_94:
                    v86 = v113;
LABEL_95:
                    v85 = v86;
                    v38 = sub_AD4C90(v6, v86, 0, v76, v77, v78);
LABEL_88:
                    if ( (_BYTE *)v134 != v136 )
                      _libc_free(v134, v85);
                    return v38;
                  }
                  v80 = *v75;
                  if ( (unsigned int)(v80 - 12) > 1 )
                  {
                    if ( (_BYTE)v80 != 17 )
                      goto LABEL_94;
                  }
                  else
                  {
                    v79 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(v6 + 8) + 24LL));
                    if ( !v79 )
                      goto LABEL_94;
                  }
                  ++v72;
                  v120 = sub_96F480(0x27u, v79, *(_QWORD *)(v74 + 8), (__int64)a3);
                  v81 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v120 + 8), v73, 0);
                  v82 = (_BYTE *)sub_96E6C0(0x19u, v120, v81, (__int64)a3);
                  v73 += v126;
                  ++v125;
                  v74 = sub_96E6C0(0x1Du, v74, v82, (__int64)a3);
                }
                while ( v116 != v125 );
                v83 = (unsigned int)v135;
                v84 = (unsigned int)v135 + 1LL;
                if ( v84 > HIDWORD(v135) )
                {
                  sub_C8D5F0(&v134, v136, v84, 8);
                  v83 = (unsigned int)v135;
                }
                ++v109;
                *(_QWORD *)(v134 + 8 * v83) = v74;
                LODWORD(v135) = v135 + 1;
                if ( v128 == v109 )
                {
LABEL_87:
                  v85 = (unsigned int)v135;
                  v38 = sub_AD3730(v134, (unsigned int)v135);
                  goto LABEL_88;
                }
              }
            }
            v121 = v128 / v124;
            v88 = sub_9208B0((__int64)a3, v43);
            v133 = v89;
            v132 = v88;
            v90 = sub_CA1930(&v132);
            v91 = 0;
            v129 = 0;
            v117 = v43;
            v108 = a2;
            if ( v68 )
            {
              v91 = v90 * (v121 - 1);
              v90 = -v90;
            }
            v107 = v91;
            v114 = v90;
            while ( 1 )
            {
              v92 = (unsigned __int8 *)sub_AD69F0(v6, v129);
              v93 = v92;
              if ( !v92 )
                goto LABEL_136;
              v94 = *v92;
              if ( (unsigned int)(v94 - 12) > 1 )
                break;
              v95 = sub_ACA8A0(v117);
              v96 = (unsigned int)v135;
              v97 = v95;
              v98 = v135;
              if ( v121 + (unsigned __int64)(unsigned int)v135 > HIDWORD(v135) )
              {
                sub_C8D5F0(&v134, v136, v121 + (unsigned __int64)(unsigned int)v135, 8);
                v96 = (unsigned int)v135;
                v98 = v135;
              }
              if ( v121 )
              {
                v99 = (__int64 *)(v134 + 8 * v96);
                v100 = &v99[v121];
                do
                  *v99++ = v97;
                while ( v100 != v99 );
                v98 = v135;
              }
              LODWORD(v135) = v121 + v98;
LABEL_116:
              if ( v124 == ++v129 )
                goto LABEL_87;
            }
            if ( (_BYTE)v94 != 17 )
            {
LABEL_136:
              v86 = v108;
              goto LABEL_95;
            }
            v101 = v107;
            v102 = 0;
            v110 = v92 + 24;
            while ( 1 )
            {
              v105 = *((_DWORD *)v93 + 8);
              v131 = v105;
              if ( v105 <= 0x40 )
                break;
              sub_C43780(&v130, v110);
              v105 = v131;
              if ( v131 <= 0x40 )
                goto LABEL_121;
              sub_C482E0(&v130, v101);
LABEL_123:
              v101 += v114;
              sub_C44740(&v132, &v130);
              v103 = sub_AD8D80(v117, &v132);
              v104 = (unsigned int)v135;
              if ( (unsigned __int64)(unsigned int)v135 + 1 > HIDWORD(v135) )
              {
                v106 = v103;
                sub_C8D5F0(&v134, v136, (unsigned int)v135 + 1LL, 8);
                v104 = (unsigned int)v135;
                v103 = v106;
              }
              *(_QWORD *)(v134 + 8 * v104) = v103;
              LODWORD(v135) = v135 + 1;
              if ( (unsigned int)v133 > 0x40 && v132 )
                j_j___libc_free_0_0(v132);
              if ( v131 > 0x40 && v130 )
                j_j___libc_free_0_0(v130);
              if ( v121 == ++v102 )
                goto LABEL_116;
            }
            v130 = *((_QWORD *)v93 + 3);
LABEL_121:
            if ( v105 == v101 )
              v130 = 0;
            else
              v130 >>= v101;
            goto LABEL_123;
          }
        }
LABEL_31:
        v40 = a2;
        return sub_AD4C90(a1, v40, 0, v12, a5, a6);
      }
LABEL_30:
      if ( (_DWORD)v12 != 18 )
        goto LABEL_31;
      goto LABEL_34;
    }
  }
  v13 = *(_QWORD *)(v9 + 24);
  v127 = *(_DWORD *)(v9 + 32);
  v14 = *(_BYTE *)(v13 + 8);
  if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
  {
    v15 = sub_BCAE30(v13);
    v135 = v16;
    v134 = v15;
    v122 = sub_CA1930(&v134);
    v18 = sub_BD5C60(a1, v9, v17);
    v19 = sub_BCCE00(v18, v122);
    v20 = sub_BCDA70(v19, v127);
    v6 = sub_AD4C90(a1, v20, 0, v21, v22, v23);
  }
  v134 = sub_9208B0((__int64)a3, a2);
  v135 = v24;
  LODWORD(v133) = sub_CA1930(&v134);
  if ( (unsigned int)v133 > 0x40 )
    sub_C43690(&v132, 0, 0);
  else
    v132 = 0;
  v134 = sub_9208B0((__int64)a3, v13);
  v135 = v25;
  v123 = sub_CA1930(&v134);
  if ( !v127 )
  {
LABEL_20:
    if ( *(_BYTE *)(a2 + 8) == 12 )
    {
      v38 = sub_AD8D80(a2, &v132);
    }
    else
    {
      v35 = sub_BCAC60(a2);
      v36 = sub_C33340();
      v37 = v36;
      if ( v35 == v36 )
        sub_C3C640(&v134, v36, &v132);
      else
        sub_C3B160(&v134, v35, &v132);
      v38 = sub_AC8EA0(*(_QWORD *)a2, &v134);
      if ( v134 == v37 )
      {
        if ( v135 )
        {
          v87 = (_QWORD *)(v135 + 24LL * *(_QWORD *)(v135 - 8));
          while ( (_QWORD *)v135 != v87 )
          {
            v87 -= 3;
            if ( v37 == *v87 )
              sub_969EE0((__int64)v87);
            else
              sub_C338F0(v87);
          }
          j_j_j___libc_free_0_0(v87 - 1);
        }
      }
      else
      {
        sub_C338F0(&v134);
      }
    }
    goto LABEL_25;
  }
  v26 = a3;
  v27 = 0;
  v28 = v26;
  while ( 1 )
  {
    if ( *v28 )
    {
      v29 = (unsigned __int8 *)sub_AD69F0(v6, v27);
      if ( !v29 )
        goto LABEL_51;
    }
    else
    {
      v29 = (unsigned __int8 *)sub_AD69F0(v6, v127 - 1 - v27);
      if ( !v29 )
        goto LABEL_51;
    }
    v30 = *v29;
    if ( (unsigned int)(v30 - 12) > 1 )
      break;
    if ( (unsigned int)v133 > 0x40 )
    {
      sub_C47690(&v132, v123);
    }
    else
    {
      v33 = 0;
      if ( v123 != (_DWORD)v133 )
        v33 = v132 << v123;
      v34 = v33 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v133);
      if ( !(_DWORD)v133 )
        v34 = 0;
      v132 = v34;
    }
LABEL_19:
    if ( v127 == ++v27 )
      goto LABEL_20;
  }
  if ( (_BYTE)v30 == 17 )
  {
    v52 = v133;
    if ( (unsigned int)v133 > 0x40 )
    {
      v115 = v29;
      sub_C47690(&v132, v123);
      v52 = v133;
      v29 = v115;
    }
    else
    {
      v53 = 0;
      if ( v123 != (_DWORD)v133 )
        v53 = v132 << v123;
      v54 = v53 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v133);
      if ( !(_DWORD)v133 )
        v54 = 0;
      v132 = v54;
    }
    sub_C449B0(&v134, v29 + 24, v52);
    if ( (unsigned int)v133 > 0x40 )
      sub_C43BD0(&v132, &v134);
    else
      v132 |= v134;
    if ( (unsigned int)v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    goto LABEL_19;
  }
LABEL_51:
  v38 = sub_AD4C90(v6, a2, 0, v30, v31, v32);
  if ( !v38 )
    goto LABEL_20;
LABEL_25:
  if ( (unsigned int)v133 > 0x40 )
  {
    if ( v132 )
      j_j___libc_free_0_0(v132);
  }
  return v38;
}
