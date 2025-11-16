// Function: sub_8D79B0
// Address: 0x8d79b0
//
__int64 __fastcall sub_8D79B0(__m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  char v5; // al
  __int8 v6; // dl
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // rax
  __m128i *v12; // r8
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rax
  const __m128i *v17; // r8
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  const __m128i *v26; // rax
  __int64 v27; // rcx
  __int8 v28; // si
  _BOOL4 v29; // edx
  __int64 v30; // r15
  __int64 v31; // rax
  __m128i *v32; // r8
  bool v33; // al
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // r15d
  __int64 v38; // rax
  __int64 *v39; // r10
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // edx
  char v44; // al
  char v45; // cl
  unsigned __int8 v46; // r11
  char v47; // di
  bool v48; // r8
  int v49; // edi
  int v50; // eax
  const __m128i *v51; // rax
  __int64 v52; // r15
  char v53; // al
  __int64 **v54; // rax
  __int64 ***v55; // rax
  _QWORD *v56; // rdi
  _QWORD *v57; // rax
  __int64 v58; // rsi
  _QWORD *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rcx
  char v63; // si
  _BOOL4 v64; // eax
  const __m128i *v65; // rax
  __int64 *v66; // r10
  _QWORD *v67; // rbx
  __int64 *v68; // r12
  __int64 **v69; // r13
  _QWORD *v70; // r14
  __int64 v71; // rax
  _QWORD *v72; // rax
  const __m128i *v73; // rdi
  const __m128i *v74; // rsi
  __int64 **v75; // rbx
  __int64 **v76; // r12
  unsigned int v77; // r13d
  __int64 v78; // rax
  __int64 *v79; // rsi
  __int64 v80; // rdi
  const __m128i *v81; // rax
  char v82; // al
  char v83; // al
  char v84; // al
  __int64 v85; // rax
  char v86; // al
  __int64 v87; // rax
  const __m128i *v88; // rax
  __int64 v89; // rax
  char v90; // al
  char v91; // al
  __int64 *v92; // rax
  __int64 *v93; // rsi
  __int64 v94; // rax
  const __m128i *v95; // rax
  __int64 *v96; // [rsp+8h] [rbp-88h]
  __int64 v97; // [rsp+10h] [rbp-80h]
  __int64 v98; // [rsp+18h] [rbp-78h]
  __int64 v99; // [rsp+18h] [rbp-78h]
  __int64 *v100; // [rsp+20h] [rbp-70h]
  __int64 v101; // [rsp+20h] [rbp-70h]
  __int64 v102; // [rsp+20h] [rbp-70h]
  __int64 *v103; // [rsp+20h] [rbp-70h]
  char v104; // [rsp+28h] [rbp-68h]
  __int64 v105; // [rsp+28h] [rbp-68h]
  const __m128i *v106; // [rsp+28h] [rbp-68h]
  __int64 v107; // [rsp+28h] [rbp-68h]
  __int64 v108; // [rsp+28h] [rbp-68h]
  __int64 *v109; // [rsp+28h] [rbp-68h]
  __int64 v110; // [rsp+28h] [rbp-68h]
  bool v111; // [rsp+37h] [rbp-59h]
  __int64 **v112; // [rsp+38h] [rbp-58h]
  __int64 v113; // [rsp+38h] [rbp-58h]
  __int64 v114; // [rsp+38h] [rbp-58h]
  bool v115; // [rsp+40h] [rbp-50h]
  __int64 v116; // [rsp+40h] [rbp-50h]
  __int64 *v117; // [rsp+48h] [rbp-48h]
  char v118; // [rsp+48h] [rbp-48h]
  __int64 v119; // [rsp+48h] [rbp-48h]
  __int64 *v120; // [rsp+50h] [rbp-40h]
  __int64 v121; // [rsp+58h] [rbp-38h]
  const __m128i *v122; // [rsp+58h] [rbp-38h]
  __int64 v123; // [rsp+58h] [rbp-38h]
  __m128i *v124; // [rsp+58h] [rbp-38h]
  __int64 ***v125; // [rsp+58h] [rbp-38h]
  const __m128i *v126; // [rsp+58h] [rbp-38h]
  __int64 v127; // [rsp+58h] [rbp-38h]
  __int64 v128; // [rsp+58h] [rbp-38h]

  v2 = a2;
  v3 = (__int64)a1;
  if ( a1 == (__m128i *)a2 )
    goto LABEL_28;
  if ( a1 )
  {
    if ( a2 )
    {
      if ( dword_4F07588 )
      {
        v4 = a1[2].m128i_i64[0];
        if ( *(_QWORD *)(a2 + 32) == v4 )
        {
          if ( v4 )
            goto LABEL_28;
        }
      }
    }
  }
  v5 = *(_BYTE *)(a2 + 140);
  v6 = a1[8].m128i_i8[12];
  if ( !dword_4F0774C )
  {
    if ( v6 == 12 )
      goto LABEL_9;
    if ( v5 == 12 )
    {
LABEL_22:
      v7 = (__int64)a1;
      goto LABEL_12;
    }
LABEL_33:
    v8 = a2;
    v7 = (__int64)a1;
LABEL_15:
    v9 = dword_4F07588;
    if ( dword_4F07588 )
    {
      v10 = *(_QWORD *)(v7 + 32);
      if ( *(_QWORD *)(v8 + 32) == v10 )
      {
        if ( v10 )
          goto LABEL_28;
      }
    }
    if ( v5 == v6 )
    {
      switch ( v5 )
      {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 9:
        case 10:
        case 11:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 21:
          v17 = (const __m128i *)v7;
          goto LABEL_43;
        case 6:
          v56 = (_QWORD *)sub_8D79B0(*(_QWORD *)(v7 + 160), *(_QWORD *)(v8 + 160));
          v57 = *(_QWORD **)(v7 + 160);
          if ( v57 == v56 )
            goto LABEL_28;
          if ( v57 )
          {
            if ( v56 )
            {
              if ( dword_4F07588 )
              {
                v58 = v56[4];
                if ( v57[4] == v58 )
                {
                  if ( v58 )
                    goto LABEL_28;
                }
              }
            }
          }
          v59 = *(_QWORD **)(v8 + 160);
          if ( v59 == v56 )
            goto LABEL_178;
          if ( v59 )
          {
            if ( v56 )
            {
              if ( dword_4F07588 )
              {
                v60 = v56[4];
                if ( v59[4] == v60 )
                {
                  if ( v60 )
                    goto LABEL_26;
                }
              }
            }
          }
          if ( (*(_BYTE *)(v7 + 168) & 1) != 0 )
            v17 = (const __m128i *)sub_72D750(v56, v7);
          else
            v17 = (const __m128i *)sub_72D2E0(v56);
          goto LABEL_43;
        case 7:
          v37 = 1;
          v120 = *(__int64 **)(v7 + 168);
          v117 = (__int64 *)*v120;
          v125 = *(__int64 ****)(v8 + 168);
          v112 = *v125;
          v38 = sub_8D79B0(*(_QWORD *)(v7 + 160), *(_QWORD *)(v8 + 160));
          v39 = v117;
          v40 = v38;
          v41 = *(_QWORD *)(v7 + 160);
          if ( v40 != v41 )
          {
            if ( v41 && v40 )
            {
              v37 = dword_4F07588;
              if ( dword_4F07588 )
                v37 = *(_QWORD *)(v40 + 32) != 0 && *(_QWORD *)(v41 + 32) == *(_QWORD *)(v40 + 32);
            }
            else
            {
              v37 = 0;
            }
          }
          v42 = *(_QWORD *)(v8 + 160);
          v43 = 1;
          if ( v40 != v42 )
          {
            if ( v40 && v42 )
            {
              v43 = dword_4F07588;
              if ( dword_4F07588 )
                v43 = *(_QWORD *)(v40 + 32) != 0 && *(_QWORD *)(v42 + 32) == *(_QWORD *)(v40 + 32);
            }
            else
            {
              v43 = 0;
            }
          }
          v44 = *((_BYTE *)v120 + 16);
          if ( (v44 & 8) != 0 )
          {
            v115 = 1;
            v45 = *((_BYTE *)v125 + 16);
          }
          else
          {
            v45 = *((_BYTE *)v125 + 16);
            v115 = (v45 & 8) != 0;
          }
          if ( ((v44 & 8) != 0) != v115 )
            v37 = 0;
          if ( v115 != ((v45 & 8) != 0) )
            v43 = 0;
          v46 = *((_BYTE *)v120 + 20);
          if ( (v46 & 1) != 0 )
          {
            v118 = 1;
            v47 = *((_BYTE *)v125 + 20);
          }
          else
          {
            v47 = *((_BYTE *)v125 + 20);
            v118 = v47 & 1;
          }
          if ( (v46 & 1) != v118 )
            v37 = 0;
          if ( v118 != (v47 & 1) )
            v43 = 0;
          v104 = v44 & 2;
          v48 = 1;
          if ( (v44 & 2) == 0 )
            v48 = (v45 & 2) != 0;
          v111 = v48;
          if ( ((v44 & 2) != 0) != v48 )
            v37 = 0;
          v49 = dword_4F077C4;
          if ( v48 != ((v45 & 2) != 0) )
            v43 = 0;
          if ( dword_4F077C4 == 2 )
          {
            if ( v120[7] )
            {
              if ( !v125[7] )
                v43 = 0;
            }
            else if ( v125[7] )
            {
              v37 = 0;
            }
            if ( *((char *)v120 + 17) < 0 )
            {
              if ( *((char *)v125 + 17) >= 0 )
                v43 = 0;
            }
            else if ( *((char *)v125 + 17) < 0 )
            {
              v37 = 0;
            }
          }
          if ( ((v46 ^ *((_BYTE *)v125 + 20)) & 8) != 0 )
          {
            if ( (*((_BYTE *)v120 + 20) & 8) != 0 )
            {
              v50 = v37;
              v43 = 0;
            }
            else
            {
              v50 = v43;
              v37 = 0;
            }
          }
          else
          {
            v50 = v43 | v37;
          }
          if ( !v50 )
            goto LABEL_189;
          if ( !v48 )
          {
            if ( v39 )
            {
              if ( v112 )
              {
                v113 = v40;
                v51 = (const __m128i *)sub_7259C0(7);
                v52 = v51[10].m128i_i64[1];
                v17 = v51;
                v51[10].m128i_i64[0] = v113;
                *(_BYTE *)(v52 + 16) = (8 * v115) | *(_BYTE *)(v52 + 16) & 0xF7;
                *(_BYTE *)(v52 + 20) = v118 | *(_BYTE *)(v52 + 20) & 0xFE;
                goto LABEL_135;
              }
              if ( v37 )
                goto LABEL_28;
              v109 = v39;
              v114 = v40;
              v88 = (const __m128i *)sub_7259C0(7);
              v52 = v88[10].m128i_i64[1];
              v17 = v88;
              v88[10].m128i_i64[0] = v114;
              *(_BYTE *)(v52 + 16) = *(_BYTE *)(v52 + 16) & 0xF7 | (8 * v115);
              *(_BYTE *)(v52 + 20) = v118 | *(_BYTE *)(v52 + 20) & 0xFE;
              v66 = v109;
              goto LABEL_254;
            }
            if ( v112 )
            {
              if ( v43 )
                goto LABEL_178;
              v108 = v40;
              v81 = (const __m128i *)sub_7259C0(7);
              v52 = v81[10].m128i_i64[1];
              v17 = v81;
              v81[10].m128i_i64[0] = v108;
              *(_BYTE *)(v52 + 16) = (8 * v115) | *(_BYTE *)(v52 + 16) & 0xF7;
              *(_BYTE *)(v52 + 20) = v118 | *(_BYTE *)(v52 + 20) & 0xFE;
              goto LABEL_226;
            }
LABEL_187:
            if ( v37 )
              goto LABEL_28;
            if ( !v43 )
            {
LABEL_189:
              v100 = v39;
              v105 = v40;
              v65 = (const __m128i *)sub_7259C0(7);
              v52 = v65[10].m128i_i64[1];
              v17 = v65;
              v65[10].m128i_i64[0] = v105;
              *(_BYTE *)(v52 + 16) = (8 * v115) | *(_BYTE *)(v52 + 16) & 0xF7;
              *(_BYTE *)(v52 + 20) = v118 | *(_BYTE *)(v52 + 20) & 0xFE;
              v66 = v100;
              if ( !v111 )
              {
                if ( v100 )
                {
                  if ( v112 )
                  {
LABEL_135:
                    v53 = (2 * v111) | *(_BYTE *)(v52 + 16) & 0xFD;
                    *(_BYTE *)(v52 + 16) = v53;
                    *(_BYTE *)(v52 + 16) = v120[2] & 1 | v53 & 0xFE;
                    if ( dword_4F077C4 == 2 )
                    {
                      *(_QWORD *)(v52 + 40) = v120[5];
                      *(_BYTE *)(v52 + 21) = *((_BYTE *)v120 + 21) & 1 | *(_BYTE *)(v52 + 21) & 0xFE;
                      *(_BYTE *)(v52 + 18) = *((_BYTE *)v120 + 18) & 0x7F | *(_BYTE *)(v52 + 18) & 0x80;
                      *(_WORD *)(v52 + 18) = *((_WORD *)v120 + 9) & 0x3F80 | *(_WORD *)(v52 + 18) & 0xC07F;
                      *(_BYTE *)(v52 + 19) = *((_BYTE *)v120 + 19) & 0xC0 | *(_BYTE *)(v52 + 19) & 0x3F;
                      v54 = (__int64 **)v120[7];
                      if ( !v54 )
                        v54 = v125[7];
                      *(_QWORD *)(v52 + 56) = v54;
                      v55 = (__int64 ***)v120;
                      if ( *((char *)v120 + 17) < 0 || (v55 = v125, *((char *)v125 + 17) < 0) )
                        *(_BYTE *)(v52 + 17) = *(_BYTE *)(v52 + 17) & 0xF | *((_BYTE *)v55 + 17) & 0x70 | 0x80;
                      v126 = v17;
                      sub_7325D0((__int64)v17, &dword_4F077C8);
                      v17 = v126;
                    }
                    goto LABEL_72;
                  }
                  goto LABEL_254;
                }
LABEL_226:
                *(_QWORD *)v52 = v112;
                goto LABEL_135;
              }
LABEL_190:
              if ( (v120[2] & 2) != 0 )
              {
                if ( ((_BYTE)v125[2] & 2) != 0 )
                {
                  if ( !v66 )
                    goto LABEL_135;
                  v116 = v7;
                  v101 = v3;
                  v67 = 0;
                  v68 = v66;
                  v98 = v2;
                  v69 = v112;
                  v119 = v8;
                  v106 = v17;
                  while ( 1 )
                  {
                    v70 = v67;
                    v71 = sub_8D8A50(v68[1], v69[1]);
                    v72 = sub_72B0C0(v71, &dword_4F077C8);
                    v73 = (const __m128i *)v68[8];
                    v74 = (const __m128i *)v69[8];
                    v67 = v72;
                    if ( v73 || v74 )
                      v72[8] = sub_5CF720(v73, v74);
                    if ( (v68[4] & 2) != 0 && ((_BYTE)v69[4] & 2) != 0 )
                      *((_BYTE *)v67 + 32) |= 2u;
                    if ( dword_4F077C4 != 2 )
                      goto LABEL_199;
                    v82 = *((_BYTE *)v68 + 32);
                    if ( (v82 & 4) != 0 )
                    {
                      v83 = *((_BYTE *)v67 + 32) | 4;
                      *((_BYTE *)v67 + 32) = v83;
                      v84 = v68[4] & 8 | v83 & 0xF7;
                      *((_BYTE *)v67 + 32) = v84;
                      *((_BYTE *)v67 + 32) = v68[4] & 0x10 | v84 & 0xEF;
                      v67[6] = v68[6];
                      v85 = v68[5];
                      if ( !v85 )
                        goto LABEL_263;
                      v67[5] = v85;
                      v67[7] = v68[7];
                      v82 = *((_BYTE *)v68 + 32);
                    }
                    else if ( ((_BYTE)v69[4] & 4) != 0 )
                    {
                      v90 = *((_BYTE *)v67 + 32) | 4;
                      *((_BYTE *)v67 + 32) = v90;
                      v91 = (_BYTE)v69[4] & 8 | v90 & 0xF7;
                      *((_BYTE *)v67 + 32) = v91;
                      *((_BYTE *)v67 + 32) = (_BYTE)v69[4] & 0x10 | v91 & 0xEF;
                      v67[6] = v69[6];
                      v92 = v69[5];
                      if ( v92 )
                      {
                        v67[5] = v92;
                        v67[7] = v69[7];
                      }
LABEL_263:
                      v82 = *((_BYTE *)v68 + 32);
                    }
                    if ( (v82 & 0x40) != 0 )
                    {
                      *((_BYTE *)v67 + 32) |= 0x40u;
                      v82 = *((_BYTE *)v68 + 32);
                    }
                    if ( v82 < 0 )
                    {
                      *((_BYTE *)v67 + 32) |= 0x80u;
                      v82 = *((_BYTE *)v68 + 32);
                    }
                    if ( (v82 & 1) != 0 )
                      *((_BYTE *)v67 + 32) |= 1u;
                    v86 = *((_BYTE *)v68 + 33);
                    if ( (v86 & 1) != 0 )
                    {
                      *((_BYTE *)v67 + 33) |= 1u;
                      v67[10] = v68[10];
                      v86 = *((_BYTE *)v68 + 33);
                    }
                    if ( (v86 & 2) != 0 )
                      *((_BYTE *)v67 + 33) |= 2u;
                    if ( dword_4F0690C )
                      *((_DWORD *)v67 + 8) = v68[4] & 0x3F800 | v67[4] & 0xFFFC07FF;
LABEL_199:
                    if ( *(_QWORD *)v52 )
                      *v70 = v67;
                    else
                      *(_QWORD *)v52 = v67;
                    v68 = (__int64 *)*v68;
                    v69 = (__int64 **)*v69;
                    if ( !v68 )
                    {
                      v8 = v119;
                      v7 = v116;
                      v17 = v106;
                      v3 = v101;
                      v2 = v98;
                      goto LABEL_135;
                    }
                  }
                }
LABEL_254:
                *(_QWORD *)v52 = v66;
                goto LABEL_135;
              }
              goto LABEL_226;
            }
LABEL_178:
            v9 = dword_4F07588;
LABEL_25:
            if ( v9 )
            {
LABEL_26:
              v11 = *(_QWORD *)(v8 + 32);
              if ( *(_QWORD *)(v7 + 32) == v11 && v11 )
                goto LABEL_28;
            }
            goto LABEL_46;
          }
          if ( (v45 & 2) != 0 )
          {
            if ( v104 )
            {
              if ( !v39 )
                goto LABEL_187;
              v107 = v7;
              v75 = (__int64 **)v39;
              v102 = v3;
              v99 = v2;
              v76 = v112;
              v77 = v43;
              while ( 1 )
              {
                if ( v75[8] )
                  v77 = 0;
                if ( v76[8] )
                  v37 = 0;
                if ( v49 == 2 )
                {
                  if ( ((_BYTE)v75[4] & 4) != 0 || v75[5] )
                  {
                    v77 = 0;
                  }
                  else if ( ((_BYTE)v76[4] & 4) != 0 )
                  {
                    v37 = 0;
                  }
                  else if ( v76[5] )
                  {
                    v37 = 0;
                  }
                }
                if ( !(v77 | v37) )
                  break;
                v96 = v39;
                v97 = v40;
                v78 = sub_8D8A50(v75[1], v76[1]);
                v79 = v75[1];
                v40 = v97;
                v39 = v96;
                if ( (__int64 *)v78 != v79 )
                {
                  if ( !v78 || !v79 || !dword_4F07588 || (v80 = *(_QWORD *)(v78 + 32), v79[4] != v80) || !v80 )
                  {
                    if ( !v77 )
                      break;
                    v37 = 0;
                  }
                }
                v93 = v76[1];
                if ( (__int64 *)v78 != v93 )
                {
                  if ( !v93 || !v78 || !dword_4F07588 || (v94 = *(_QWORD *)(v78 + 32), v93[4] != v94) || !v94 )
                  {
                    if ( !v37 )
                      break;
                    v77 = 0;
                  }
                }
                v75 = (__int64 **)*v75;
                v76 = (__int64 **)*v76;
                if ( !v75 )
                {
                  v43 = v77;
                  v7 = v107;
                  v3 = v102;
                  v2 = v99;
                  goto LABEL_187;
                }
                v49 = dword_4F077C4;
              }
              v7 = v107;
              v3 = v102;
              v2 = v99;
            }
            else if ( v43 )
            {
              goto LABEL_178;
            }
          }
          else if ( v37 )
          {
            goto LABEL_28;
          }
          v103 = v39;
          v110 = v40;
          v95 = (const __m128i *)sub_7259C0(7);
          v52 = v95[10].m128i_i64[1];
          v17 = v95;
          v95[10].m128i_i64[0] = v110;
          *(_BYTE *)(v52 + 16) = (8 * v115) | *(_BYTE *)(v52 + 16) & 0xF7;
          *(_BYTE *)(v52 + 20) = v118 | *(_BYTE *)(v52 + 20) & 0xFE;
          v66 = v103;
          goto LABEL_190;
        case 8:
          if ( (*(_WORD *)(v7 + 168) & 0x180) == 0 && *(_QWORD *)(v7 + 176) )
          {
            v127 = *(_QWORD *)(v7 + 176);
            v87 = sub_8D79B0(*(_QWORD *)(v7 + 160), *(_QWORD *)(v8 + 160));
            v20 = v127;
            v21 = v87;
          }
          else if ( (*(_WORD *)(v8 + 168) & 0x180) == 0 && *(_QWORD *)(v8 + 176) )
          {
            v128 = *(_QWORD *)(v8 + 176);
            v89 = sub_8D79B0(*(_QWORD *)(v8 + 160), *(_QWORD *)(v7 + 160));
            v20 = v128;
            v21 = v89;
          }
          else
          {
            v18 = *(_BYTE *)(v7 + 169);
            if ( (v18 & 0x10) != 0 )
              goto LABEL_28;
            if ( (*(_BYTE *)(v8 + 169) & 0x10) != 0 )
              goto LABEL_25;
            if ( (v18 & 2) != 0 )
              goto LABEL_28;
            if ( (*(_BYTE *)(v8 + 169) & 2) != 0 )
              goto LABEL_25;
            if ( *(char *)(v7 + 168) < 0 )
              goto LABEL_28;
            if ( *(char *)(v8 + 168) < 0 )
              goto LABEL_25;
            v19 = sub_8D79B0(*(_QWORD *)(v7 + 160), *(_QWORD *)(v8 + 160));
            v20 = 0;
            v21 = v19;
          }
          v22 = *(_QWORD *)(v7 + 160);
          if ( v22 != v21
            && (!v21 || !v22 || !dword_4F07588 || (v23 = *(_QWORD *)(v21 + 32), *(_QWORD *)(v22 + 32) != v23) || !v23)
            || (*(_BYTE *)(v7 + 169) & 1) != 0
            || *(_QWORD *)(v7 + 176) != v20 )
          {
            v24 = *(_QWORD *)(v8 + 160);
            if ( v24 == v21
              || v24 && v21 && dword_4F07588 && (v25 = *(_QWORD *)(v21 + 32), *(_QWORD *)(v24 + 32) == v25) && v25 )
            {
              if ( (*(_BYTE *)(v8 + 169) & 1) == 0 && *(_QWORD *)(v8 + 176) == v20 )
                goto LABEL_178;
            }
            v121 = v20;
            v26 = (const __m128i *)sub_7259C0(8);
            v26[10].m128i_i64[0] = v21;
            v26[11].m128i_i64[0] = v121;
            v122 = v26;
            sub_8D6090((__int64)v26);
            v17 = v122;
LABEL_72:
            if ( (const __m128i *)v7 != v17 )
              goto LABEL_73;
          }
          goto LABEL_28;
        case 13:
          v30 = sub_8D4870(v7);
          v123 = sub_8D4870(v8);
          v31 = sub_8D79B0(v30, v123);
          v32 = (__m128i *)v31;
          if ( v30 == v31 )
            goto LABEL_28;
          v33 = v31 != 0;
          if ( v30 )
          {
            if ( v33 )
            {
              if ( dword_4F07588 )
              {
                v34 = v32[2].m128i_i64[0];
                if ( *(_QWORD *)(v30 + 32) == v34 )
                {
                  if ( v34 )
                    goto LABEL_28;
                }
              }
            }
          }
          if ( (__m128i *)v123 == v32 )
          {
            v17 = (const __m128i *)v8;
            goto LABEL_44;
          }
          if ( !v123 || !v33 || !dword_4F07588 || (v35 = v32[2].m128i_i64[0], *(_QWORD *)(v123 + 32) != v35) || !v35 )
          {
            v124 = v32;
            v36 = sub_8D4890(v7);
            v17 = (const __m128i *)sub_73F0A0(v124, v36);
            goto LABEL_43;
          }
          if ( v8 )
            goto LABEL_26;
          goto LABEL_46;
        case 14:
          if ( !**(_QWORD **)(v7 + 168) && **(_QWORD **)(v8 + 168) )
            goto LABEL_25;
          goto LABEL_28;
        default:
          sub_721090();
      }
    }
    v17 = (const __m128i *)sub_72C930();
LABEL_43:
    if ( v17 == (const __m128i *)v7 )
      goto LABEL_28;
LABEL_44:
    if ( !v17 )
    {
      if ( !v8 )
      {
LABEL_46:
        v12 = (__m128i *)v2;
        goto LABEL_29;
      }
LABEL_75:
      v12 = sub_73CA70(v17, v3);
LABEL_29:
      if ( !HIDWORD(qword_4F077B4) )
        return (__int64)v12;
      v6 = *(_BYTE *)(v3 + 140);
      v5 = *(_BYTE *)(v2 + 140);
      if ( v6 != 12 )
      {
LABEL_37:
        if ( v5 != 12 )
          goto LABEL_40;
        goto LABEL_38;
      }
LABEL_35:
      v14 = v3;
      do
      {
        v14 = *(_QWORD *)(v14 + 160);
        v6 = *(_BYTE *)(v14 + 140);
      }
      while ( v6 == 12 );
      goto LABEL_37;
    }
LABEL_73:
    if ( !dword_4F07588 )
    {
      if ( (const __m128i *)v8 == v17 )
        goto LABEL_46;
      goto LABEL_75;
    }
    v61 = v17[2].m128i_i64[0];
    if ( *(_QWORD *)(v7 + 32) != v61 || !v61 )
    {
      if ( (const __m128i *)v8 == v17 || dword_4F07588 && *(_QWORD *)(v8 + 32) == v61 && v61 )
        goto LABEL_46;
      goto LABEL_75;
    }
LABEL_28:
    v12 = (__m128i *)v3;
    goto LABEL_29;
  }
  if ( v6 == 12 )
  {
    if ( v5 == 14 && !*(_BYTE *)(a2 + 160) )
    {
      v27 = *(_QWORD *)(a2 + 168);
      v7 = (__int64)a1;
      if ( *(_DWORD *)(v27 + 28) != -1 )
      {
        do
        {
LABEL_10:
          v7 = *(_QWORD *)(v7 + 160);
          v6 = *(_BYTE *)(v7 + 140);
        }
        while ( v6 == 12 );
        if ( v5 != 12 )
        {
          v8 = v2;
          goto LABEL_14;
        }
LABEL_12:
        v8 = v2;
        do
        {
          v8 = *(_QWORD *)(v8 + 160);
          v5 = *(_BYTE *)(v8 + 140);
        }
        while ( v5 == 12 );
LABEL_14:
        if ( v7 == v8 )
          goto LABEL_28;
        goto LABEL_15;
      }
      v28 = a1[11].m128i_i8[8];
      v29 = v28 == 2;
      if ( *(_DWORD *)(v27 + 24) == 1 )
        v29 = v28 == 3;
      if ( v29 )
      {
        v12 = a1;
        if ( !HIDWORD(qword_4F077B4) )
          return (__int64)v12;
        goto LABEL_35;
      }
    }
LABEL_9:
    v7 = (__int64)a1;
    goto LABEL_10;
  }
  if ( v5 != 12 )
    goto LABEL_33;
  if ( v6 != 14 || a1[10].m128i_i8[0] )
    goto LABEL_22;
  v62 = a1[10].m128i_i64[1];
  v7 = (__int64)a1;
  if ( *(_DWORD *)(v62 + 28) != -1 )
    goto LABEL_12;
  v63 = *(_BYTE *)(a2 + 184);
  v64 = v63 == 2;
  if ( *(_DWORD *)(v62 + 24) == 1 )
    v64 = v63 == 3;
  if ( !v64 )
    goto LABEL_22;
  v12 = (__m128i *)v2;
  if ( !HIDWORD(qword_4F077B4) )
    return (__int64)v12;
LABEL_38:
  v15 = v2;
  do
  {
    v15 = *(_QWORD *)(v15 + 160);
    v5 = *(_BYTE *)(v15 + 140);
  }
  while ( v5 == 12 );
LABEL_40:
  if ( v6 != v5 )
    return (__int64)v12;
  v16 = sub_5D1620((__int64)v12, v3);
  return sub_5D1620(v16, v2);
}
