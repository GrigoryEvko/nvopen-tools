// Function: sub_96AF60
// Address: 0x96af60
//
__int64 __fastcall sub_96AF60(__int64 a1, _QWORD *a2, unsigned __int8 **a3, __int64 a4)
{
  unsigned int v4; // r15d
  unsigned __int8 *v6; // rbx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int8 *v11; // rsi
  unsigned __int8 *v12; // rdx
  unsigned __int8 *v13; // rsi
  unsigned __int64 *v14; // rax
  __int64 v15; // rax
  __int64 result; // rax
  int v17; // edx
  __int64 v18; // rsi
  unsigned __int8 *v19; // rdx
  int v20; // eax
  unsigned __int8 *v21; // r12
  unsigned __int8 *v22; // rdx
  int v23; // eax
  unsigned __int8 *v24; // r12
  int v25; // eax
  __int64 v26; // r14
  unsigned __int8 *v27; // rdx
  int v28; // eax
  unsigned __int8 *v29; // rdi
  int v30; // eax
  unsigned __int8 *v31; // rdi
  unsigned __int8 *v32; // r13
  int v33; // ebx
  int v34; // eax
  int v35; // r9d
  int v36; // ebx
  int v37; // edx
  unsigned int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rsi
  unsigned __int64 v41; // rsi
  int v42; // edx
  __int64 v43; // rax
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  unsigned __int8 *v46; // rax
  unsigned __int8 *v47; // rax
  __int64 v48; // r14
  unsigned __int8 *v49; // rsi
  unsigned __int8 *v50; // rbx
  int v51; // eax
  unsigned __int8 *v52; // rsi
  int v53; // edx
  unsigned __int8 *v54; // rax
  unsigned __int8 *v55; // rsi
  _QWORD *v56; // rcx
  unsigned int v57; // r13d
  unsigned int v58; // r14d
  int v59; // eax
  unsigned __int64 v60; // rdi
  unsigned int v61; // esi
  __int64 v62; // rdx
  __int64 v63; // rdi
  __int64 v64; // rdx
  int v65; // ecx
  bool v66; // zf
  unsigned __int64 v67; // rax
  int v68; // eax
  __int64 v69; // rbx
  unsigned __int8 *v70; // r14
  int v71; // eax
  unsigned __int8 *v72; // r13
  int v73; // eax
  unsigned __int8 *v74; // r13
  int v75; // r15d
  unsigned int v76; // r14d
  unsigned int v77; // eax
  __int64 v78; // rsi
  __int64 v79; // rdi
  __int64 v80; // rdx
  unsigned int v81; // edx
  unsigned __int8 *v82; // r12
  int v83; // eax
  unsigned __int8 *v84; // rdi
  __int64 v85; // rax
  __int16 v86; // ax
  unsigned __int8 *v87; // rsi
  __int64 v88; // r13
  unsigned __int8 *v89; // rdx
  unsigned __int8 *v90; // rsi
  int v91; // esi
  char v92; // r8
  __int64 v93; // r14
  __int64 v94; // r13
  int v95; // eax
  __int64 v96; // rsi
  __int64 v97; // rcx
  unsigned __int8 v98; // al
  char v99; // al
  unsigned int v100; // r14d
  unsigned __int64 v101; // rbx
  int v102; // eax
  __int64 v103; // rdx
  unsigned __int64 *v104; // rbx
  __int64 v105; // r14
  unsigned int v106; // eax
  __int64 v107; // rsi
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rax
  unsigned int v110; // eax
  unsigned int v111; // ebx
  __int64 v112; // r15
  unsigned __int64 v113; // rax
  unsigned __int64 *v114; // rsi
  unsigned int v115; // ebx
  unsigned __int64 *v116; // rsi
  int v117; // eax
  unsigned __int8 v118; // al
  char v119; // al
  __int64 v120; // rax
  unsigned __int64 *v121; // rbx
  _QWORD *v122; // r14
  unsigned __int64 *v123; // r14
  _QWORD *v124; // r14
  _QWORD *v125; // r14
  _QWORD *v126; // rbx
  _QWORD *v127; // rbx
  _QWORD *v128; // rbx
  unsigned __int64 v129; // rdi
  int v130; // eax
  __int64 *v131; // rsi
  unsigned __int64 v132; // rax
  unsigned __int64 v133; // rax
  unsigned __int8 v134; // al
  char v135; // al
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  _QWORD *v139; // rbx
  _QWORD *v140; // rbx
  _QWORD *v141; // r14
  _QWORD *v142; // r14
  _QWORD *v143; // r14
  _QWORD *v144; // r14
  _QWORD *v145; // r14
  unsigned __int64 *v146; // r14
  _QWORD *v147; // rdi
  _QWORD *v148; // rbx
  _QWORD *v149; // r14
  _QWORD *v150; // r14
  unsigned __int64 v151; // rax
  unsigned __int64 v152; // rax
  unsigned __int64 v153; // rax
  unsigned __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  unsigned __int64 *v157; // r14
  _QWORD *v158; // rbx
  _QWORD *v159; // r12
  unsigned __int64 v160; // rbx
  _QWORD *v161; // rdx
  _QWORD *v162; // rbx
  _QWORD *v163; // rdx
  __int64 v164; // r12
  unsigned __int8 *v165; // r13
  _QWORD *v166; // rbx
  _QWORD *v167; // rdx
  __int64 v168; // r12
  unsigned __int8 *v169; // r13
  _QWORD *v170; // rbx
  __int64 v171; // rax
  _QWORD *v172; // r14
  _QWORD *v173; // rdx
  _QWORD *v174; // r12
  _QWORD *v175; // r14
  unsigned __int8 *v176; // [rsp+8h] [rbp-218h]
  __int64 v177; // [rsp+20h] [rbp-200h]
  bool v178; // [rsp+28h] [rbp-1F8h]
  unsigned int v179; // [rsp+28h] [rbp-1F8h]
  __int64 *v180; // [rsp+30h] [rbp-1F0h]
  __int64 *v181; // [rsp+38h] [rbp-1E8h]
  __int64 *v182; // [rsp+40h] [rbp-1E0h]
  unsigned __int8 v183; // [rsp+70h] [rbp-1B0h]
  unsigned __int8 *v184; // [rsp+78h] [rbp-1A8h]
  char v185; // [rsp+78h] [rbp-1A8h]
  int v186; // [rsp+78h] [rbp-1A8h]
  int v187; // [rsp+78h] [rbp-1A8h]
  unsigned __int8 *v188; // [rsp+78h] [rbp-1A8h]
  unsigned __int8 *v189; // [rsp+80h] [rbp-1A0h]
  bool v190; // [rsp+80h] [rbp-1A0h]
  unsigned int v191; // [rsp+80h] [rbp-1A0h]
  unsigned __int8 *v192; // [rsp+80h] [rbp-1A0h]
  unsigned int v193; // [rsp+80h] [rbp-1A0h]
  unsigned int v194; // [rsp+80h] [rbp-1A0h]
  __int64 v195; // [rsp+80h] [rbp-1A0h]
  unsigned int v196; // [rsp+80h] [rbp-1A0h]
  unsigned int v197; // [rsp+80h] [rbp-1A0h]
  __int64 v199; // [rsp+88h] [rbp-198h]
  __int64 v200; // [rsp+88h] [rbp-198h]
  __int64 v201; // [rsp+88h] [rbp-198h]
  __int64 v202; // [rsp+88h] [rbp-198h]
  __int64 v203; // [rsp+88h] [rbp-198h]
  __int64 v204; // [rsp+88h] [rbp-198h]
  __int64 v205; // [rsp+88h] [rbp-198h]
  __int64 v206; // [rsp+88h] [rbp-198h]
  __int64 v207; // [rsp+90h] [rbp-190h] BYREF
  _QWORD *v208; // [rsp+98h] [rbp-188h]
  __int64 v209; // [rsp+B0h] [rbp-170h] BYREF
  _QWORD *v210; // [rsp+B8h] [rbp-168h]
  __int64 v211; // [rsp+D0h] [rbp-150h] BYREF
  _QWORD *v212; // [rsp+D8h] [rbp-148h]
  __int64 v213; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v214; // [rsp+F8h] [rbp-128h]
  char v215; // [rsp+104h] [rbp-11Ch]
  __int64 v216; // [rsp+110h] [rbp-110h] BYREF
  _QWORD *v217; // [rsp+118h] [rbp-108h]
  __int64 v218; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v219; // [rsp+138h] [rbp-E8h]
  char v220; // [rsp+144h] [rbp-DCh]
  __int64 v221; // [rsp+150h] [rbp-D0h] BYREF
  _QWORD *v222; // [rsp+158h] [rbp-C8h]
  unsigned __int64 v223; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v224; // [rsp+178h] [rbp-A8h]
  char v225; // [rsp+184h] [rbp-9Ch]
  __int64 v226; // [rsp+190h] [rbp-90h] BYREF
  _QWORD *v227; // [rsp+198h] [rbp-88h]
  unsigned __int64 v228; // [rsp+1B0h] [rbp-70h] BYREF
  __int64 v229; // [rsp+1B8h] [rbp-68h]
  char v230; // [rsp+1C4h] [rbp-5Ch]
  unsigned __int64 v231; // [rsp+1D0h] [rbp-50h] BYREF
  unsigned __int64 *v232; // [rsp+1D8h] [rbp-48h]

  v4 = a1;
  v6 = *a3;
  if ( **a3 != 18 )
    goto LABEL_29;
  v189 = a3[1];
  if ( *v189 != 18 || (v184 = a3[2], *v184 != 18) )
  {
    if ( (unsigned int)(a1 - 331) > 1 )
    {
      if ( (unsigned int)(a1 - 180) <= 1 )
      {
        v25 = *v6;
        goto LABEL_64;
      }
LABEL_22:
      if ( v4 == 3071 )
      {
        v68 = **a3;
        v69 = (__int64)(*a3 + 24);
        if ( (_BYTE)v68 != 17 )
        {
          if ( (unsigned int)(v68 - 12) > 1 )
            return 0;
          v69 = 0;
        }
        v70 = a3[1];
        v71 = *v70;
        if ( (_BYTE)v71 == 17 )
        {
          v192 = v70 + 24;
        }
        else
        {
          v192 = 0;
          if ( (unsigned int)(v71 - 12) > 1 )
            return 0;
        }
        v72 = a3[2];
        v73 = *v72;
        if ( (_BYTE)v73 == 17 )
        {
          v74 = v72 + 24;
          v75 = 0;
          v76 = 0;
          LODWORD(v232) = 32;
          v231 = 0;
          do
          {
            v77 = sub_C44320(v74, 8, v76);
            v78 = 255;
            if ( v77 <= 0xC )
            {
              v78 = 0;
              if ( v77 != 12 )
              {
                v79 = v69;
                if ( (v77 & 0xA) != 0xA )
                {
                  v79 = (__int64)v192;
                  if ( (v77 & 0xC) == 4 )
                    v79 = v69;
                }
                if ( v79 )
                {
                  if ( v77 <= 7 )
                    v78 = (unsigned int)sub_C44320(v79, 8, 8 * (v77 & 3));
                  else
                    v78 = 255 * (unsigned int)sub_C44320(v79, 1, (v77 & 1) == 0 ? 15 : 31);
                }
                else
                {
                  ++v75;
                  v78 = 0;
                }
              }
            }
            v80 = v76;
            v76 += 8;
            sub_C43FB0(&v231, v78, v80, 8);
          }
          while ( v76 != 32 );
          if ( v75 == 4 )
          {
            result = sub_ACA8A0(a2);
            goto LABEL_128;
          }
          goto LABEL_127;
        }
        if ( (unsigned int)(v73 - 12) <= 1 )
          return sub_ACA8A0(a2);
        return 0;
      }
      result = 0;
      if ( v4 != 8999 )
        return result;
      v17 = **a3;
      v18 = (__int64)(*a3 + 24);
      if ( (_BYTE)v17 != 17 )
      {
        if ( (unsigned int)(v17 - 12) > 1 )
          return result;
        v18 = 0;
      }
      v19 = a3[1];
      v20 = *v19;
      if ( (_BYTE)v20 == 17 )
      {
        v21 = a3[2];
        v22 = v19 + 24;
        v23 = *v21;
        if ( (_BYTE)v23 == 17 )
        {
          v24 = v21 + 24;
          if ( v18 )
          {
            sub_C472A0(&v228, v18, v22);
            sub_C45EE0(&v228, v24);
            v102 = v229;
            LODWORD(v229) = 0;
            LODWORD(v232) = v102;
            v231 = v228;
            result = sub_AD8D80(a2, &v231);
            goto LABEL_57;
          }
          return sub_AD8D80(a2, v24);
        }
        if ( (unsigned int)(v23 - 12) > 1 )
          return 0;
        if ( v18 )
        {
          sub_C472A0(&v231, v18, v22);
          goto LABEL_127;
        }
      }
      else
      {
        if ( (unsigned int)(v20 - 12) > 1 )
          return 0;
        v82 = a3[2];
        v83 = *v82;
        v24 = v82 + 24;
        if ( (_BYTE)v83 == 17 )
          return sub_AD8D80(a2, v24);
        if ( (unsigned int)(v83 - 12) > 1 )
          return 0;
      }
      return sub_AD6530(a2);
    }
    if ( *v189 != 13 )
      return 0;
    return sub_ACADE0(a2);
  }
  if ( *(_BYTE *)a4 == 85 )
  {
    v85 = *(_QWORD *)(a4 - 32);
    if ( v85 )
    {
      if ( !*(_BYTE *)v85 && *(_QWORD *)(v85 + 24) == *(_QWORD *)(a4 + 80) && (*(_BYTE *)(v85 + 33) & 0x20) != 0 )
      {
        a1 = a4;
        if ( (unsigned __int8)sub_B5A1B0(a4) )
        {
          v86 = sub_B59DB0(a4);
          v183 = v86;
          if ( HIBYTE(v86) != 1 || (_BYTE)v86 == 7 )
            v183 = 1;
          v87 = v6 + 24;
          v88 = sub_C33340();
          if ( *((_QWORD *)v6 + 3) == v88 )
            sub_C3C790(&v231, v87);
          else
            sub_C33EB0(&v231, v87);
          result = 0;
          if ( ((v4 - 107) & 0xFFFFFFFD) == 0 )
          {
            v89 = v184 + 24;
            v90 = v189 + 24;
            v91 = v88 == v231 ? sub_C3F220(&v231, v90, v89, v183) : sub_C3B3E0(&v231, v90, v89, v183);
            v92 = sub_968D50(a4, v91);
            result = 0;
            if ( v92 )
              result = sub_AC8EA0(*a2, &v231);
          }
          if ( v88 == v231 )
          {
            if ( v232 )
            {
              v103 = 3 * *(v232 - 1);
              v104 = &v232[v103];
              if ( v232 != &v232[v103] )
              {
                v105 = result;
                do
                {
                  v104 -= 3;
                  if ( v88 == *v104 )
                    sub_969EE0((__int64)v104);
                  else
                    sub_C338F0(v104);
                }
                while ( v232 != v104 );
                result = v105;
              }
              v206 = result;
              j_j_j___libc_free_0_0(v104 - 1);
              return v206;
            }
          }
          else
          {
            v205 = result;
            sub_C338F0(&v231);
            return v205;
          }
          return result;
        }
      }
    }
  }
  if ( v4 > 0x82C )
  {
    if ( v4 == 2205 )
    {
      v45 = sub_C33340();
      v9 = *((_QWORD *)v6 + 3);
      v10 = v45;
      v46 = v6 + 24;
      if ( v9 == v10 )
        v46 = (unsigned __int8 *)*((_QWORD *)v6 + 4);
      if ( (v46[20] & 7) == 3
        || (v10 == *((_QWORD *)v189 + 3) ? (v47 = (unsigned __int8 *)*((_QWORD *)v189 + 4)) : (v47 = v189 + 24),
            (v47[20] & 7) == 3) )
      {
        v48 = sub_C33310(a1, a2);
        sub_C3B170(&v231, 0.0);
        sub_C407B0(&v228, &v231, v48);
        sub_C338F0(&v231);
        if ( v10 == v228 )
          sub_C3C790(&v231, &v228);
        else
          sub_C33EB0(&v231, &v228);
        v49 = v184 + 24;
        if ( v10 == v231 )
          sub_C3D800(&v231, v49, 1);
        else
          sub_C3ADF0(&v231, v49, 1);
        v202 = sub_AC8EA0(*a2, &v231);
        sub_91D830(&v231);
        sub_91D830(&v228);
        return v202;
      }
LABEL_9:
      v11 = v6 + 24;
      if ( v10 == v9 )
        sub_C3C790(&v231, v11);
      else
        sub_C33EB0(&v231, v11);
      v12 = v184 + 24;
      v13 = v189 + 24;
      if ( v231 == v10 )
        sub_C3F220(&v231, v13, v12, 1);
      else
        sub_C3B3E0(&v231, v13, v12, 1);
      v14 = &v231;
      if ( v231 == v10 )
        v14 = v232;
      if ( (*((_BYTE *)v14 + 20) & 7) == 1 && (unsigned __int8)sub_C33750() )
      {
        v15 = 0;
        goto LABEL_18;
      }
LABEL_17:
      v15 = sub_AC8EA0(*a2, &v231);
LABEL_18:
      v199 = v15;
      sub_91D830(&v231);
      return v199;
    }
LABEL_30:
    if ( v4 - 180 <= 1 )
    {
      v25 = **a3;
      if ( (_BYTE)v25 == 17 )
      {
        v26 = (__int64)(*a3 + 24);
LABEL_33:
        v27 = a3[1];
        v28 = *v27;
        if ( (_BYTE)v28 == 17 )
        {
          v29 = a3[2];
          v30 = *v29;
          if ( (_BYTE)v30 == 17 )
          {
            v185 = 0;
            v31 = v29 + 24;
            v32 = v27 + 24;
            v190 = v26 == 0;
            goto LABEL_36;
          }
LABEL_137:
          v81 = v30 - 12;
          result = 0;
          if ( v81 > 1 )
            return result;
          return (__int64)a3[v4 == 181];
        }
        if ( (unsigned int)(v28 - 12) <= 1 )
        {
          v84 = a3[2];
          v30 = *v84;
          if ( (_BYTE)v30 != 17 )
            goto LABEL_137;
          v31 = v84 + 24;
          if ( v26 )
          {
            v190 = 0;
            v32 = 0;
            v185 = 1;
LABEL_36:
            v33 = *((_DWORD *)v31 + 2);
            v34 = sub_C459C0();
            v35 = v34;
            if ( v34 )
            {
              v36 = v33 - v34;
              v37 = v36;
              if ( v4 != 181 )
              {
                v35 = v36;
                v37 = v34;
              }
              if ( v190 )
              {
                v110 = *((_DWORD *)v32 + 2);
                LODWORD(v232) = v110;
                if ( v110 > 0x40 )
                {
                  v196 = v35;
                  sub_C43780(&v231, v32);
                  v110 = (unsigned int)v232;
                  v35 = v196;
                  if ( (unsigned int)v232 > 0x40 )
                  {
                    sub_C482E0(&v231, v196);
                    goto LABEL_127;
                  }
                }
                else
                {
                  v231 = *(_QWORD *)v32;
                }
                if ( v110 == v35 )
                  v231 = 0;
                else
                  v231 >>= v35;
                goto LABEL_127;
              }
              if ( !v185 )
              {
                v38 = *((_DWORD *)v32 + 2);
                LODWORD(v229) = v38;
                if ( v38 > 0x40 )
                {
                  v187 = v37;
                  v193 = v35;
                  sub_C43780(&v228, v32);
                  v38 = v229;
                  v35 = v193;
                  v37 = v187;
                  if ( (unsigned int)v229 > 0x40 )
                  {
                    sub_C482E0(&v228, v193);
                    v37 = v187;
                    goto LABEL_45;
                  }
                }
                else
                {
                  v228 = *(_QWORD *)v32;
                }
                if ( v38 == v35 )
                  v228 = 0;
                else
                  v228 >>= v35;
LABEL_45:
                v39 = *(_DWORD *)(v26 + 8);
                LODWORD(v232) = v39;
                if ( v39 > 0x40 )
                {
                  v194 = v37;
                  sub_C43780(&v231, v26);
                  v39 = (unsigned int)v232;
                  v37 = v194;
                  if ( (unsigned int)v232 > 0x40 )
                  {
                    sub_C47690(&v231, v194);
LABEL_52:
                    v42 = v229;
                    if ( (unsigned int)v229 > 0x40 )
                    {
                      sub_C43BD0(&v228, &v231);
                      v42 = v229;
                      v43 = v228;
                    }
                    else
                    {
                      v43 = v231 | v228;
                      v228 |= v231;
                    }
                    LODWORD(v227) = v42;
                    v226 = v43;
                    LODWORD(v229) = 0;
                    result = sub_AD8D80(a2, &v226);
                    if ( (unsigned int)v227 > 0x40 && v226 )
                    {
                      v200 = result;
                      j_j___libc_free_0_0(v226);
                      result = v200;
                    }
LABEL_57:
                    if ( (unsigned int)v232 > 0x40 && v231 )
                    {
                      v201 = result;
                      j_j___libc_free_0_0(v231);
                      result = v201;
                    }
                    if ( (unsigned int)v229 <= 0x40 )
                      return result;
                    v44 = v228;
                    if ( !v228 )
                      return result;
LABEL_110:
                    v204 = result;
                    j_j___libc_free_0_0(v44);
                    return v204;
                  }
                }
                else
                {
                  v231 = *(_QWORD *)v26;
                }
                v40 = 0;
                if ( v37 != v39 )
                  v40 = v231 << v37;
                v41 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v39) & v40;
                if ( !v39 )
                  v41 = 0;
                v231 = v41;
                goto LABEL_52;
              }
              v106 = *(_DWORD *)(v26 + 8);
              LODWORD(v232) = v106;
              if ( v106 <= 0x40 )
              {
                v231 = *(_QWORD *)v26;
LABEL_241:
                v107 = 0;
                if ( v37 != v106 )
                  v107 = v231 << v37;
                v108 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v106;
                v66 = v106 == 0;
                v109 = 0;
                if ( !v66 )
                  v109 = v108;
                v231 = v109 & v107;
                goto LABEL_127;
              }
              v197 = v37;
              sub_C43780(&v231, v26);
              v106 = (unsigned int)v232;
              v37 = v197;
              if ( (unsigned int)v232 <= 0x40 )
                goto LABEL_241;
              sub_C47690(&v231, v197);
LABEL_127:
              result = sub_AD8D80(a2, &v231);
LABEL_128:
              if ( (unsigned int)v232 <= 0x40 )
                return result;
              v44 = v231;
              if ( !v231 )
                return result;
              goto LABEL_110;
            }
            return (__int64)a3[v4 == 181];
          }
          return sub_ACA8A0(a2);
        }
        return 0;
      }
LABEL_64:
      if ( (unsigned int)(v25 - 12) > 1 )
        return 0;
      v26 = 0;
      goto LABEL_33;
    }
    goto LABEL_22;
  }
  if ( v4 > 0x828 )
  {
    v93 = *((_QWORD *)v6 + 3);
    v177 = v93;
    v94 = sub_C33340();
    if ( v93 == v94 )
    {
      sub_C3C460(&v207, v93);
      sub_C3C460(&v209, v93);
      sub_C3C460(&v211, v93);
    }
    else
    {
      sub_C37380(&v207, v93);
      sub_C37380(&v209, v93);
      sub_C37380(&v211, v93);
    }
    v181 = (__int64 *)(v6 + 24);
    if ( *((_QWORD *)v6 + 3) == v94 )
      sub_C3C790(&v228, v181);
    else
      sub_C33EB0(&v228, v6 + 24);
    if ( v94 == v228 )
    {
      if ( (*(_BYTE *)(v229 + 20) & 8) == 0 )
        goto LABEL_252;
      sub_C3CCB0(&v228);
      v132 = v228;
    }
    else
    {
      if ( (v230 & 8) == 0 )
      {
LABEL_172:
        sub_C338E0(&v231, &v228);
        goto LABEL_173;
      }
      sub_C34440(&v228);
      v132 = v228;
    }
    if ( v94 != v132 )
      goto LABEL_172;
LABEL_252:
    sub_C3C840(&v231, &v228);
LABEL_173:
    v182 = (__int64 *)(v184 + 24);
    if ( *((_QWORD *)v184 + 3) == v94 )
      sub_C3C790(&v223, v182);
    else
      sub_C33EB0(&v223, v184 + 24);
    if ( v94 == v223 )
    {
      if ( (*(_BYTE *)(v224 + 20) & 8) == 0 )
        goto LABEL_250;
      sub_C3CCB0(&v223);
      v133 = v223;
    }
    else
    {
      if ( (v225 & 8) == 0 )
      {
LABEL_177:
        sub_C338E0(&v226, &v223);
        goto LABEL_178;
      }
      sub_C34440(&v223);
      v133 = v223;
    }
    if ( v133 != v94 )
      goto LABEL_177;
LABEL_250:
    sub_C3C840(&v226, &v223);
LABEL_178:
    if ( v226 == v94 )
      v95 = sub_C3E510(&v226, &v231);
    else
      v95 = sub_C37950(&v226, &v231);
    v178 = 0;
    v180 = (__int64 *)(v189 + 24);
    if ( (unsigned int)(v95 - 1) > 1 )
      goto LABEL_181;
    if ( v94 == *((_QWORD *)v189 + 3) )
      sub_C3C790(&v218, v180);
    else
      sub_C33EB0(&v218, v189 + 24);
    if ( v94 == v218 )
    {
      if ( (*(_BYTE *)(v219 + 20) & 8) == 0 )
        goto LABEL_514;
      sub_C3CCB0(&v218);
      v155 = v218;
    }
    else
    {
      if ( (v220 & 8) == 0 )
      {
LABEL_416:
        sub_C338E0(&v221, &v218);
        goto LABEL_417;
      }
      sub_C34440(&v218);
      v155 = v218;
    }
    if ( v94 != v155 )
      goto LABEL_416;
LABEL_514:
    sub_C3C840(&v221, &v218);
LABEL_417:
    if ( v94 == *((_QWORD *)v184 + 3) )
      sub_C3C790(&v213, v182);
    else
      sub_C33EB0(&v213, v182);
    if ( v94 == v213 )
    {
      if ( (*(_BYTE *)(v214 + 20) & 8) == 0 )
        goto LABEL_512;
      sub_C3CCB0(&v213);
      v156 = v213;
    }
    else
    {
      if ( (v215 & 8) == 0 )
      {
LABEL_421:
        sub_C338E0(&v216, &v213);
LABEL_422:
        if ( v94 == v216 )
          v130 = sub_C3E510(&v216, &v221);
        else
          v130 = sub_C37950(&v216, &v221);
        v178 = (unsigned int)(v130 - 1) <= 1;
        if ( v94 == v216 )
        {
          if ( v217 )
          {
            v147 = &v217[3 * *(v217 - 1)];
            if ( v217 != v147 )
            {
              v176 = v6;
              v148 = &v217[3 * *(v217 - 1)];
              do
              {
                v148 -= 3;
                if ( v94 == *v148 )
                  sub_969EE0((__int64)v148);
                else
                  sub_C338F0(v148);
              }
              while ( v217 != v148 );
              v147 = v148;
              v6 = v176;
            }
            j_j_j___libc_free_0_0(v147 - 1);
          }
        }
        else
        {
          sub_C338F0(&v216);
        }
        if ( v94 == v213 )
        {
          if ( v214 )
          {
            v150 = (_QWORD *)(v214 + 24LL * *(_QWORD *)(v214 - 8));
            while ( (_QWORD *)v214 != v150 )
            {
              v150 -= 3;
              if ( v94 == *v150 )
                sub_969EE0((__int64)v150);
              else
                sub_C338F0(v150);
            }
            j_j_j___libc_free_0_0(v150 - 1);
          }
        }
        else
        {
          sub_C338F0(&v213);
        }
        if ( v94 == v221 )
        {
          if ( v222 )
          {
            v149 = &v222[3 * *(v222 - 1)];
            while ( v222 != v149 )
            {
              v149 -= 3;
              if ( v94 == *v149 )
                sub_969EE0((__int64)v149);
              else
                sub_C338F0(v149);
            }
            j_j_j___libc_free_0_0(v149 - 1);
          }
        }
        else
        {
          sub_C338F0(&v221);
        }
        if ( v94 == v218 )
        {
          if ( v219 )
          {
            v144 = (_QWORD *)(v219 + 24LL * *(_QWORD *)(v219 - 8));
            while ( (_QWORD *)v219 != v144 )
            {
              v144 -= 3;
              if ( v94 == *v144 )
                sub_969EE0((__int64)v144);
              else
                sub_C338F0(v144);
            }
            j_j_j___libc_free_0_0(v144 - 1);
          }
        }
        else
        {
          sub_C338F0(&v218);
        }
LABEL_181:
        if ( v94 == v226 )
        {
          if ( v227 )
          {
            v125 = &v227[3 * *(v227 - 1)];
            while ( v227 != v125 )
            {
              v125 -= 3;
              if ( v94 == *v125 )
                sub_969EE0((__int64)v125);
              else
                sub_C338F0(v125);
            }
            j_j_j___libc_free_0_0(v125 - 1);
          }
        }
        else
        {
          sub_C338F0(&v226);
        }
        if ( v94 == v223 )
        {
          if ( v224 )
          {
            v124 = (_QWORD *)(v224 + 24LL * *(_QWORD *)(v224 - 8));
            while ( (_QWORD *)v224 != v124 )
            {
              v124 -= 3;
              if ( v94 == *v124 )
                sub_969EE0((__int64)v124);
              else
                sub_C338F0(v124);
            }
            j_j_j___libc_free_0_0(v124 - 1);
          }
        }
        else
        {
          sub_C338F0(&v223);
        }
        if ( v94 == v231 )
        {
          if ( v232 )
          {
            v123 = &v232[3 * *(v232 - 1)];
            while ( v232 != v123 )
            {
              v123 -= 3;
              if ( v94 == *v123 )
                sub_969EE0((__int64)v123);
              else
                sub_C338F0(v123);
            }
            j_j_j___libc_free_0_0(v123 - 1);
          }
        }
        else
        {
          sub_C338F0(&v231);
        }
        if ( v94 == v228 )
        {
          if ( v229 )
          {
            v122 = (_QWORD *)(v229 + 24LL * *(_QWORD *)(v229 - 8));
            while ( (_QWORD *)v229 != v122 )
            {
              v122 -= 3;
              if ( v94 == *v122 )
                sub_969EE0((__int64)v122);
              else
                sub_C338F0(v122);
            }
            j_j_j___libc_free_0_0(v122 - 1);
          }
        }
        else
        {
          sub_C338F0(&v228);
        }
        v96 = *((_QWORD *)v6 + 3);
        if ( v178 )
        {
          v97 = *((_QWORD *)v184 + 3);
          if ( v94 == v97 )
          {
            v98 = *(_BYTE *)(*((_QWORD *)v184 + 4) + 20LL);
            if ( (v98 & 8) == 0 )
              goto LABEL_194;
          }
          else
          {
            v98 = v184[44];
            if ( (v98 & 8) == 0 )
              goto LABEL_194;
          }
          v99 = v98 & 7;
          if ( v99 != 3 && v99 != 1 )
          {
            if ( v94 == v96 )
              sub_C3C790(&v231, v181);
            else
              sub_C33EB0(&v231, v181);
            if ( v94 == v231 )
              sub_C3CCB0(&v231);
            else
              sub_C34440(&v231);
            v100 = 5;
            sub_96AAC0(&v209, (__int64 *)&v231);
            sub_91D830(&v231);
            v97 = *((_QWORD *)v184 + 3);
            goto LABEL_197;
          }
LABEL_194:
          if ( v94 != v209 )
          {
            if ( v94 != v96 )
            {
              v100 = 4;
              sub_C33E70(&v209, v181);
              v97 = *((_QWORD *)v184 + 3);
              goto LABEL_197;
            }
            if ( v181 != &v209 )
            {
              sub_C338F0(&v209);
LABEL_410:
              if ( v94 != *((_QWORD *)v6 + 3) )
              {
                sub_C33EB0(&v209, v181);
                v100 = 4;
                v97 = *((_QWORD *)v184 + 3);
                goto LABEL_197;
              }
              sub_C3C790(&v209, v181);
              v97 = *((_QWORD *)v184 + 3);
              goto LABEL_448;
            }
            goto LABEL_448;
          }
          if ( v94 != v96 )
          {
            if ( v181 != &v209 )
            {
              if ( v210 )
              {
                v141 = &v210[3 * *(v210 - 1)];
                while ( v210 != v141 )
                {
                  v141 -= 3;
                  if ( v94 == *v141 )
                    sub_969EE0((__int64)v141);
                  else
                    sub_C338F0(v141);
                }
                j_j_j___libc_free_0_0(v141 - 1);
              }
              goto LABEL_410;
            }
LABEL_448:
            v100 = 4;
            goto LABEL_197;
          }
          v100 = 4;
          sub_C3C9E0(&v209, v181);
          v97 = *((_QWORD *)v184 + 3);
LABEL_197:
          if ( v94 == v207 )
          {
            if ( v94 == v97 )
            {
              sub_C3C9E0(&v207, v182);
            }
            else if ( v182 != &v207 )
            {
              if ( v208 )
              {
                v139 = &v208[3 * *(v208 - 1)];
                while ( v208 != v139 )
                {
                  v139 -= 3;
                  if ( v94 == *v139 )
                    sub_969EE0((__int64)v139);
                  else
                    sub_C338F0(v139);
                }
                j_j_j___libc_free_0_0(v139 - 1);
              }
              goto LABEL_396;
            }
          }
          else
          {
            if ( v94 != v97 )
            {
              sub_C33E70(&v207, v182);
              goto LABEL_200;
            }
            if ( v182 != &v207 )
            {
              sub_C338F0(&v207);
LABEL_396:
              if ( v94 == *((_QWORD *)v184 + 3) )
                sub_C3C790(&v207, v182);
              else
                sub_C33EB0(&v207, v182);
            }
          }
LABEL_200:
          if ( v94 == *((_QWORD *)v189 + 3) )
            sub_C3C790(&v231, v180);
          else
            sub_C33EB0(&v231, v180);
          if ( v94 == v231 )
            sub_C3CCB0(&v231);
          else
            sub_C34440(&v231);
          v101 = v231;
          if ( v94 == v211 )
          {
            if ( v94 == v231 )
            {
              if ( v212 )
              {
                v173 = &v212[3 * *(v212 - 1)];
                if ( v173 != v212 )
                {
                  v174 = &v212[3 * *(v212 - 1)];
                  do
                  {
                    v174 -= 3;
                    if ( *v174 == v101 )
                      sub_969EE0((__int64)v174);
                    else
                      sub_C338F0(v174);
                  }
                  while ( v212 != v174 );
                  v173 = v174;
                }
                j_j_j___libc_free_0_0(v173 - 1);
              }
              goto LABEL_730;
            }
            if ( v212 )
            {
              v140 = &v212[3 * *(v212 - 1)];
              while ( v212 != v140 )
              {
                v140 -= 3;
                if ( v94 == *v140 )
                  sub_969EE0((__int64)v140);
                else
                  sub_C338F0(v140);
              }
              j_j_j___libc_free_0_0(v140 - 1);
            }
          }
          else
          {
            if ( v94 != v231 )
            {
              sub_C33870(&v211, &v231);
              goto LABEL_207;
            }
            sub_C338F0(&v211);
          }
          if ( v94 != v231 )
          {
            sub_C338E0(&v211, &v231);
            goto LABEL_207;
          }
LABEL_730:
          sub_C3C840(&v211, &v231);
LABEL_207:
          if ( v94 != v231 )
          {
LABEL_208:
            sub_C338F0(&v231);
            goto LABEL_209;
          }
          if ( v232 )
          {
            v121 = &v232[3 * *(v232 - 1)];
            while ( v232 != v121 )
            {
              v121 -= 3;
              if ( v94 == *v121 )
                sub_969EE0((__int64)v121);
              else
                sub_C338F0(v121);
            }
LABEL_471:
            j_j_j___libc_free_0_0(v121 - 1);
            goto LABEL_209;
          }
          goto LABEL_209;
        }
        if ( v94 == v96 )
          sub_C3C790(&v228, v181);
        else
          sub_C33EB0(&v228, v181);
        if ( v94 == v228 )
        {
          if ( (*(_BYTE *)(v229 + 20) & 8) == 0 )
            goto LABEL_479;
          sub_C3CCB0(&v228);
          v154 = v228;
        }
        else
        {
          if ( (v230 & 8) == 0 )
          {
LABEL_292:
            sub_C338E0(&v231, &v228);
            goto LABEL_293;
          }
          sub_C34440(&v228);
          v154 = v228;
        }
        if ( v94 != v154 )
          goto LABEL_292;
LABEL_479:
        sub_C3C840(&v231, &v228);
LABEL_293:
        if ( v94 == *((_QWORD *)v189 + 3) )
          sub_C3C790(&v223, v180);
        else
          sub_C33EB0(&v223, v180);
        if ( v94 == v223 )
        {
          if ( (*(_BYTE *)(v224 + 20) & 8) == 0 )
            goto LABEL_477;
          sub_C3CCB0(&v223);
          v153 = v223;
        }
        else
        {
          if ( (v225 & 8) == 0 )
          {
LABEL_297:
            sub_C338E0(&v226, &v223);
            goto LABEL_298;
          }
          sub_C34440(&v223);
          v153 = v223;
        }
        if ( v94 != v153 )
          goto LABEL_297;
LABEL_477:
        sub_C3C840(&v226, &v223);
LABEL_298:
        if ( v94 == v226 )
          v117 = sub_C3E510(&v226, &v231);
        else
          v117 = sub_C37950(&v226, &v231);
        v179 = v117 - 1;
        if ( v94 == v226 )
        {
          if ( v227 )
          {
            v143 = &v227[3 * *(v227 - 1)];
            while ( v227 != v143 )
            {
              v143 -= 3;
              if ( v94 == *v143 )
                sub_969EE0((__int64)v143);
              else
                sub_C338F0(v143);
            }
            j_j_j___libc_free_0_0(v143 - 1);
          }
        }
        else
        {
          sub_C338F0(&v226);
        }
        if ( v94 == v223 )
        {
          if ( v224 )
          {
            v142 = (_QWORD *)(v224 + 24LL * *(_QWORD *)(v224 - 8));
            while ( (_QWORD *)v224 != v142 )
            {
              v142 -= 3;
              if ( v94 == *v142 )
                sub_969EE0((__int64)v142);
              else
                sub_C338F0(v142);
            }
            j_j_j___libc_free_0_0(v142 - 1);
          }
        }
        else
        {
          sub_C338F0(&v223);
        }
        if ( v94 == v231 )
        {
          if ( v232 )
          {
            v146 = &v232[3 * *(v232 - 1)];
            while ( v232 != v146 )
            {
              v146 -= 3;
              if ( v94 == *v146 )
                sub_969EE0((__int64)v146);
              else
                sub_C338F0(v146);
            }
            j_j_j___libc_free_0_0(v146 - 1);
          }
        }
        else
        {
          sub_C338F0(&v231);
        }
        if ( v94 == v228 )
        {
          if ( v229 )
          {
            v145 = (_QWORD *)(v229 + 24LL * *(_QWORD *)(v229 - 8));
            while ( (_QWORD *)v229 != v145 )
            {
              v145 -= 3;
              if ( v94 == *v145 )
                sub_969EE0((__int64)v145);
              else
                sub_C338F0(v145);
            }
            j_j_j___libc_free_0_0(v145 - 1);
          }
        }
        else
        {
          sub_C338F0(&v228);
        }
        if ( v179 > 1 )
        {
          if ( v94 == *((_QWORD *)v6 + 3) )
          {
            v118 = *(_BYTE *)(*((_QWORD *)v6 + 4) + 20LL);
            if ( (v118 & 8) == 0 )
              goto LABEL_313;
          }
          else
          {
            v118 = v6[44];
            if ( (v118 & 8) == 0 )
              goto LABEL_313;
          }
          v119 = v118 & 7;
          if ( v119 == 3 || v119 == 1 )
          {
LABEL_313:
            if ( v94 == *((_QWORD *)v184 + 3) )
              sub_C3C790(&v231, v182);
            else
              sub_C33EB0(&v231, v182);
            if ( v94 == v231 )
              sub_C3CCB0(&v231);
            else
              sub_C34440(&v231);
            sub_96AAC0(&v209, (__int64 *)&v231);
            if ( v94 == v231 )
            {
              if ( v232 )
              {
                v157 = &v232[3 * *(v232 - 1)];
                while ( v232 != v157 )
                {
                  v157 -= 3;
                  if ( v94 == *v157 )
                    sub_969EE0((__int64)v157);
                  else
                    sub_C338F0(v157);
                }
                j_j_j___libc_free_0_0(v157 - 1);
              }
            }
            else
            {
              sub_C338F0(&v231);
            }
            v100 = 0;
LABEL_320:
            v120 = *((_QWORD *)v6 + 3);
            if ( v94 == v207 )
            {
              if ( v94 == v120 )
              {
                sub_C3C9E0(&v207, v181);
              }
              else if ( v181 != &v207 )
              {
                if ( v208 )
                {
                  v161 = &v208[3 * *(v208 - 1)];
                  if ( v208 != v161 )
                  {
                    v188 = v6;
                    v162 = &v208[3 * *(v208 - 1)];
                    do
                    {
                      v162 -= 3;
                      if ( v94 == *v162 )
                        sub_969EE0((__int64)v162);
                      else
                        sub_C338F0(v162);
                    }
                    while ( v208 != v162 );
                    v161 = v162;
                    v6 = v188;
                  }
                  j_j_j___libc_free_0_0(v161 - 1);
                }
                goto LABEL_595;
              }
            }
            else
            {
              if ( v94 != v120 )
              {
                sub_C33E70(&v207, v181);
                goto LABEL_323;
              }
              if ( v181 != &v207 )
              {
                sub_C338F0(&v207);
LABEL_595:
                if ( v94 == *((_QWORD *)v6 + 3) )
                  sub_C3C790(&v207, v181);
                else
                  sub_C33EB0(&v207, v181);
              }
            }
LABEL_323:
            if ( v94 == *((_QWORD *)v189 + 3) )
              sub_C3C790(&v231, v180);
            else
              sub_C33EB0(&v231, v180);
            if ( v94 == v231 )
              sub_C3CCB0(&v231);
            else
              sub_C34440(&v231);
            if ( v94 == v211 )
            {
              if ( v94 == v231 )
              {
                if ( v212 )
                {
                  v158 = &v212[3 * *(v212 - 1)];
                  if ( v212 != v158 )
                  {
                    v159 = &v212[3 * *(v212 - 1)];
                    v160 = v231;
                    do
                    {
                      v159 -= 3;
                      if ( *v159 == v160 )
                        sub_969EE0((__int64)v159);
                      else
                        sub_C338F0(v159);
                    }
                    while ( v212 != v159 );
                    v158 = v159;
                  }
                  j_j_j___libc_free_0_0(v158 - 1);
                }
                goto LABEL_745;
              }
            }
            else if ( v94 != v231 )
            {
              sub_C33870(&v211, &v231);
              goto LABEL_330;
            }
            sub_91D830(&v211);
            if ( v94 != v231 )
            {
              sub_C338E0(&v211, &v231);
              goto LABEL_330;
            }
LABEL_745:
            sub_C3C840(&v211, &v231);
LABEL_330:
            if ( v94 != v231 )
              goto LABEL_208;
            if ( v232 )
            {
              v121 = &v232[3 * *(v232 - 1)];
              while ( v232 != v121 )
              {
                v121 -= 3;
                if ( v94 == *v121 )
                  sub_969EE0((__int64)v121);
                else
                  sub_C338F0(v121);
              }
              goto LABEL_471;
            }
            goto LABEL_209;
          }
          v171 = *((_QWORD *)v184 + 3);
          if ( v94 == v209 )
          {
            if ( v94 == v171 )
            {
              v100 = 1;
              sub_C3C9E0(&v209, v182);
              goto LABEL_320;
            }
            if ( v182 != &v209 )
            {
              if ( v210 )
              {
                v175 = &v210[3 * *(v210 - 1)];
                while ( v210 != v175 )
                {
                  v175 -= 3;
                  if ( v94 == *v175 )
                    sub_969EE0((__int64)v175);
                  else
                    sub_C338F0(v175);
                }
                j_j_j___libc_free_0_0(v175 - 1);
              }
              goto LABEL_739;
            }
          }
          else
          {
            if ( v94 != v171 )
            {
              v100 = 1;
              sub_C33E70(&v209, v182);
              goto LABEL_320;
            }
            if ( v182 != &v209 )
            {
              sub_C338F0(&v209);
LABEL_739:
              if ( v94 != *((_QWORD *)v184 + 3) )
              {
                sub_C33EB0(&v209, v182);
                v100 = 1;
                goto LABEL_320;
              }
              sub_C3C790(&v209, v182);
            }
          }
          v100 = 1;
          goto LABEL_320;
        }
        if ( v94 == *((_QWORD *)v189 + 3) )
        {
          v134 = *(_BYTE *)(*((_QWORD *)v189 + 4) + 20LL);
          if ( (v134 & 8) == 0 )
            goto LABEL_484;
        }
        else
        {
          v134 = v189[44];
          if ( (v134 & 8) == 0 )
            goto LABEL_484;
        }
        v135 = v134 & 7;
        if ( v135 != 3 && v135 != 1 )
        {
          if ( v94 == *((_QWORD *)v184 + 3) )
            sub_C3C790(&v231, v182);
          else
            sub_C33EB0(&v231, v182);
          if ( v94 == v231 )
            sub_C3CCB0(&v231);
          else
            sub_C34440(&v231);
          v100 = 3;
          sub_96AAC0(&v211, (__int64 *)&v231);
          sub_91D830(&v231);
LABEL_487:
          v137 = *((_QWORD *)v189 + 3);
          if ( v94 == v207 )
          {
            if ( v94 == v137 )
            {
              sub_C3C9E0(&v207, v180);
              goto LABEL_490;
            }
            if ( v180 == &v207 )
              goto LABEL_490;
            if ( v208 )
            {
              v167 = &v208[3 * *(v208 - 1)];
              if ( v208 != v167 )
              {
                v168 = v94;
                v169 = v6;
                v170 = &v208[3 * *(v208 - 1)];
                do
                {
                  v170 -= 3;
                  if ( v168 == *v170 )
                    sub_969EE0((__int64)v170);
                  else
                    sub_C338F0(v170);
                }
                while ( v208 != v170 );
                v167 = v170;
                v6 = v169;
                v94 = v168;
              }
              j_j_j___libc_free_0_0(v167 - 1);
            }
          }
          else
          {
            if ( v94 != v137 )
            {
              sub_C33E70(&v207, v180);
              goto LABEL_490;
            }
            if ( v180 == &v207 )
            {
LABEL_490:
              v138 = *((_QWORD *)v6 + 3);
              if ( v94 == v209 )
              {
                if ( v94 == v138 )
                {
                  sub_C3C9E0(&v209, v181);
                  goto LABEL_209;
                }
                if ( v181 == &v209 )
                  goto LABEL_209;
                if ( v210 )
                {
                  v163 = &v210[3 * *(v210 - 1)];
                  if ( v210 != v163 )
                  {
                    v164 = v94;
                    v165 = v6;
                    v166 = &v210[3 * *(v210 - 1)];
                    do
                    {
                      v166 -= 3;
                      if ( v164 == *v166 )
                        sub_969EE0((__int64)v166);
                      else
                        sub_C338F0(v166);
                    }
                    while ( v210 != v166 );
                    v163 = v166;
                    v6 = v165;
                    v94 = v164;
                  }
                  j_j_j___libc_free_0_0(v163 - 1);
                }
              }
              else
              {
                if ( v94 != v138 )
                {
                  sub_C33E70(&v209, v181);
                  goto LABEL_209;
                }
                if ( v181 == &v209 )
                {
LABEL_209:
                  if ( v4 == 2091 )
                  {
                    v131 = &v209;
                    if ( v94 == v209 )
                    {
LABEL_439:
                      sub_C3C840(&v231, v131);
                      goto LABEL_214;
                    }
                  }
                  else
                  {
                    if ( v4 != 2092 )
                    {
                      if ( v4 == 2090 )
                      {
                        if ( v94 == v207 )
                          sub_C3C790(&v231, &v207);
                        else
                          sub_C33EB0(&v231, &v207);
                        if ( v94 == v231 )
                          sub_C3D800(&v231, &v207, 1);
                        else
                          sub_C3ADF0(&v231, &v207, 1);
                      }
                      else if ( v177 == v94 )
                      {
                        sub_C3C5A0(&v231, v94, v100);
                      }
                      else
                      {
                        sub_C36740(&v231, v177, v100);
                      }
                      goto LABEL_214;
                    }
                    v131 = &v211;
                    if ( v94 == v211 )
                      goto LABEL_439;
                  }
                  sub_C338E0(&v231, v131);
LABEL_214:
                  if ( v94 == v211 )
                  {
                    if ( v212 )
                    {
                      v128 = &v212[3 * *(v212 - 1)];
                      while ( v212 != v128 )
                      {
                        v128 -= 3;
                        if ( v94 == *v128 )
                          sub_969EE0((__int64)v128);
                        else
                          sub_C338F0(v128);
                      }
                      j_j_j___libc_free_0_0(v128 - 1);
                    }
                  }
                  else
                  {
                    sub_C338F0(&v211);
                  }
                  if ( v94 == v209 )
                  {
                    if ( v210 )
                    {
                      v127 = &v210[3 * *(v210 - 1)];
                      while ( v210 != v127 )
                      {
                        v127 -= 3;
                        if ( v94 == *v127 )
                          sub_969EE0((__int64)v127);
                        else
                          sub_C338F0(v127);
                      }
                      j_j_j___libc_free_0_0(v127 - 1);
                    }
                  }
                  else
                  {
                    sub_C338F0(&v209);
                  }
                  if ( v94 == v207 )
                  {
                    if ( v208 )
                    {
                      v126 = &v208[3 * *(v208 - 1)];
                      while ( v208 != v126 )
                      {
                        v126 -= 3;
                        if ( v94 == *v126 )
                          sub_969EE0((__int64)v126);
                        else
                          sub_C338F0(v126);
                      }
                      j_j_j___libc_free_0_0(v126 - 1);
                    }
                  }
                  else
                  {
                    sub_C338F0(&v207);
                  }
                  goto LABEL_17;
                }
                sub_C338F0(&v209);
              }
              if ( v94 == *((_QWORD *)v6 + 3) )
                sub_C3C790(&v209, v181);
              else
                sub_C33EB0(&v209, v181);
              goto LABEL_209;
            }
            sub_C338F0(&v207);
          }
          if ( v94 == *((_QWORD *)v189 + 3) )
            sub_C3C790(&v207, v180);
          else
            sub_C33EB0(&v207, v180);
          goto LABEL_490;
        }
LABEL_484:
        v136 = *((_QWORD *)v184 + 3);
        if ( v94 == v211 )
        {
          if ( v94 == v136 )
          {
            v100 = 2;
            sub_C3C9E0(&v211, v182);
            goto LABEL_487;
          }
          if ( v182 != &v211 )
          {
            if ( v212 )
            {
              v172 = &v212[3 * *(v212 - 1)];
              while ( v212 != v172 )
              {
                v172 -= 3;
                if ( v94 == *v172 )
                  sub_969EE0((__int64)v172);
                else
                  sub_C338F0(v172);
              }
              j_j_j___libc_free_0_0(v172 - 1);
            }
            goto LABEL_648;
          }
        }
        else
        {
          if ( v94 != v136 )
          {
            v100 = 2;
            sub_C33E70(&v211, v182);
            goto LABEL_487;
          }
          if ( v182 != &v211 )
          {
            sub_C338F0(&v211);
LABEL_648:
            if ( v94 == *((_QWORD *)v184 + 3) )
              sub_C3C790(&v211, v182);
            else
              sub_C33EB0(&v211, v182);
            v100 = 2;
            goto LABEL_487;
          }
        }
        v100 = 2;
        goto LABEL_487;
      }
      sub_C34440(&v213);
      v156 = v213;
    }
    if ( v94 != v156 )
      goto LABEL_421;
LABEL_512:
    sub_C3C840(&v216, &v213);
    goto LABEL_422;
  }
  if ( v4 - 173 <= 1 )
  {
    v8 = sub_C33340();
    v9 = *((_QWORD *)v6 + 3);
    v10 = v8;
    goto LABEL_9;
  }
LABEL_29:
  if ( v4 - 331 > 1 )
    goto LABEL_30;
  v50 = *a3;
  v51 = **a3;
  if ( (_BYTE)v51 == 13 )
    return sub_ACADE0(a2);
  v52 = a3[1];
  v53 = *v52;
  if ( (_BYTE)v53 == 13 )
    return sub_ACADE0(a2);
  if ( (_BYTE)v51 != 17 )
  {
    if ( (unsigned int)(v51 - 12) > 1 )
      return 0;
    if ( (_BYTE)v53 == 17 )
      return sub_AD6530(a2);
LABEL_133:
    if ( (unsigned int)(v53 - 12) > 1 )
      return 0;
    return sub_AD6530(a2);
  }
  if ( (_BYTE)v53 != 17 )
    goto LABEL_133;
  v54 = a3[2];
  v55 = v52 + 24;
  v56 = (_QWORD *)*((_QWORD *)v54 + 3);
  if ( *((_DWORD *)v54 + 8) > 0x40u )
    v56 = (_QWORD *)*v56;
  v57 = *((_DWORD *)v50 + 8);
  v186 = (int)v56;
  v191 = (unsigned int)v56;
  v58 = 2 * v57;
  sub_C44830(&v228, v55, 2 * v57);
  sub_C44830(&v231, v50 + 24, 2 * v57);
  sub_C472A0(&v226, &v231, &v228);
  v59 = (int)v227;
  LODWORD(v224) = (_DWORD)v227;
  if ( (unsigned int)v227 > 0x40 )
  {
    sub_C43780(&v223, &v226);
    v59 = v224;
    if ( (unsigned int)v224 > 0x40 )
    {
      sub_C44B70(&v223, v191);
      v61 = (unsigned int)v227;
      goto LABEL_95;
    }
    v60 = v223;
    v61 = (unsigned int)v227;
  }
  else
  {
    v60 = v226;
    v61 = (unsigned int)v227;
  }
  v62 = 0;
  if ( v59 )
    v62 = (__int64)(v60 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59);
  v63 = v62 >> 63;
  v64 = v62 >> v186;
  if ( v186 != v59 )
    v63 = v64;
  v65 = -v59;
  v66 = v59 == 0;
  v67 = 0;
  if ( !v66 )
    v67 = v63 & (0xFFFFFFFFFFFFFFFFLL >> v65);
  v223 = v67;
LABEL_95:
  if ( v61 > 0x40 && v226 )
    j_j___libc_free_0_0(v226);
  if ( (unsigned int)v232 > 0x40 && v231 )
    j_j___libc_free_0_0(v231);
  if ( (unsigned int)v229 > 0x40 && v228 )
    j_j___libc_free_0_0(v228);
  if ( v4 == 332 )
  {
    v111 = v57 - 1;
    LODWORD(v232) = v57;
    v112 = 1LL << ((unsigned __int8)v57 - 1);
    if ( v57 <= 0x40 )
    {
      v113 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v57;
      if ( !v57 )
        v113 = 0;
      v231 = ~(1LL << ((unsigned __int8)v57 - 1)) & v113;
      sub_C44830(&v226, &v231, v58);
      if ( (unsigned int)v232 <= 0x40 )
      {
        LODWORD(v232) = v57;
LABEL_268:
        v231 = 0;
        goto LABEL_269;
      }
LABEL_589:
      v129 = v231;
      if ( !v231 )
      {
LABEL_402:
        LODWORD(v232) = v57;
        if ( v57 <= 0x40 )
          goto LABEL_268;
LABEL_403:
        sub_C43690(&v231, 0, 0);
        if ( (unsigned int)v232 > 0x40 )
        {
          *(_QWORD *)(v231 + 8LL * (v111 >> 6)) |= v112;
          goto LABEL_270;
        }
LABEL_269:
        v231 |= v112;
LABEL_270:
        sub_C44830(&v228, &v231, v58);
        if ( (unsigned int)v232 > 0x40 && v231 )
          j_j___libc_free_0_0(v231);
        v114 = (unsigned __int64 *)&v226;
        if ( (int)sub_C4C880(&v223, &v226) < 0 )
          v114 = &v223;
        if ( (unsigned int)v224 <= 0x40 && (v115 = *((_DWORD *)v114 + 2), v115 <= 0x40) )
        {
          v151 = *v114;
          LODWORD(v224) = *((_DWORD *)v114 + 2);
          v223 = v151;
          v116 = &v223;
          if ( (int)sub_C4C880(&v223, &v228) > 0 )
            goto LABEL_599;
          v116 = &v228;
        }
        else
        {
          sub_C43990(&v223, v114);
          v116 = &v228;
          if ( (int)sub_C4C880(&v223, &v228) > 0 )
            v116 = &v223;
          if ( (unsigned int)v224 > 0x40 )
            goto LABEL_281;
        }
        v115 = *((_DWORD *)v116 + 2);
        if ( v115 > 0x40 )
        {
LABEL_281:
          sub_C43990(&v223, v116);
          goto LABEL_282;
        }
LABEL_599:
        v152 = *v116;
        LODWORD(v224) = v115;
        v223 = v152;
LABEL_282:
        if ( (unsigned int)v229 > 0x40 && v228 )
          j_j___libc_free_0_0(v228);
        if ( (unsigned int)v227 > 0x40 && v226 )
          j_j___libc_free_0_0(v226);
        goto LABEL_105;
      }
LABEL_401:
      j_j___libc_free_0_0(v129);
      goto LABEL_402;
    }
    v195 = ~(1LL << ((unsigned __int8)v57 - 1));
    sub_C43690(&v231, -1, 1);
    if ( (unsigned int)v232 <= 0x40 )
    {
      v231 &= v195;
      sub_C44830(&v226, &v231, v58);
      if ( (unsigned int)v232 > 0x40 )
        goto LABEL_589;
    }
    else
    {
      *(_QWORD *)(v231 + 8LL * (v111 >> 6)) &= v195;
      sub_C44830(&v226, &v231, v58);
      if ( (unsigned int)v232 > 0x40 )
      {
        v129 = v231;
        if ( v231 )
          goto LABEL_401;
      }
    }
    LODWORD(v232) = v57;
    goto LABEL_403;
  }
LABEL_105:
  sub_C44B10(&v231, &v223, v57);
  result = sub_ACCFD0(*a2, &v231);
  if ( (unsigned int)v232 > 0x40 && v231 )
  {
    v203 = result;
    j_j___libc_free_0_0(v231);
    result = v203;
  }
  if ( (unsigned int)v224 > 0x40 )
  {
    v44 = v223;
    if ( v223 )
      goto LABEL_110;
  }
  return result;
}
