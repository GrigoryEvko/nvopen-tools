// Function: sub_83EB80
// Address: 0x83eb80
//
void __fastcall sub_83EB80(
        __int64 *a1,
        __int64 a2,
        const __m128i *a3,
        int a4,
        _BOOL4 a5,
        int a6,
        int a7,
        __int64 a8,
        int a9,
        int a10,
        __int64 *a11)
{
  __int64 v11; // rbx
  int v12; // eax
  _QWORD *v13; // r14
  __int64 v14; // r15
  char v15; // bl
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v18; // r13
  __int64 v19; // r10
  __m128i *v20; // rax
  bool v21; // zf
  const __m128i *v22; // r12
  int v23; // eax
  __int64 v24; // r10
  int v25; // ecx
  char v26; // al
  bool v27; // dl
  __m128i *v28; // rdi
  int v29; // eax
  __int64 v30; // r10
  __int8 v31; // dl
  int v32; // ecx
  __int64 *v33; // rdi
  char v34; // r12
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // dl
  __m128i v39; // xmm1
  char v40; // dl
  char v41; // r12
  __int64 v42; // rdi
  unsigned int v43; // eax
  int v44; // eax
  __m128i *i; // rdx
  int v46; // eax
  const __m128i *v47; // rdi
  __m128i *v48; // rdx
  __m128i *v49; // rdx
  __int8 v50; // al
  __m128i *v51; // rsi
  __int64 j; // rcx
  const __m128i *k; // rdi
  __int64 v54; // r8
  __int8 v55; // al
  int v56; // eax
  int v57; // eax
  __int64 v58; // rax
  int v59; // eax
  __int64 v60; // rdi
  __int64 v61; // rax
  int v62; // eax
  __m128i *v63; // rax
  __m128i *v64; // rdx
  int v65; // eax
  __m128i *v66; // rax
  int v67; // eax
  __int64 v68; // r10
  unsigned __int64 v69; // rcx
  __int64 *v70; // rsi
  char v71; // al
  int v72; // eax
  int v73; // eax
  __int64 v74; // rax
  int v75; // eax
  char v76; // al
  __m128i *v77; // rax
  int v78; // eax
  __m128i *v79; // rax
  __m128i *v80; // rdi
  unsigned int v81; // r8d
  int v82; // eax
  unsigned int v83; // eax
  __int64 v84; // rsi
  __int64 v85; // r8
  __int64 v86; // rax
  __int64 v87; // rax
  int v88; // eax
  int v89; // eax
  __m128i *v90; // rax
  char v91; // al
  unsigned __int8 *v92; // rax
  __m128i *v93; // rax
  int v94; // eax
  __int64 v95; // rax
  char v96; // al
  __int64 *v97; // rsi
  __int64 *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // rdi
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 v105; // r8
  int v106; // eax
  __m128i *v107; // rax
  __int64 v108; // r10
  int v109; // eax
  unsigned int v110; // eax
  int v111; // r12d
  int v112; // eax
  __m128i *v113; // rax
  unsigned int v114; // [rsp+4h] [rbp-15Ch]
  int v115; // [rsp+4h] [rbp-15Ch]
  int v117; // [rsp+Ch] [rbp-154h]
  int v121; // [rsp+28h] [rbp-138h]
  int v122; // [rsp+28h] [rbp-138h]
  bool v123; // [rsp+28h] [rbp-138h]
  int v124; // [rsp+28h] [rbp-138h]
  __int64 v125; // [rsp+28h] [rbp-138h]
  __int64 v126; // [rsp+30h] [rbp-130h]
  int v127; // [rsp+30h] [rbp-130h]
  __int64 v128; // [rsp+30h] [rbp-130h]
  __int64 v129; // [rsp+30h] [rbp-130h]
  __int64 v130; // [rsp+30h] [rbp-130h]
  __int64 v131; // [rsp+30h] [rbp-130h]
  __int64 v132; // [rsp+30h] [rbp-130h]
  __int64 v133; // [rsp+30h] [rbp-130h]
  __int64 v134; // [rsp+30h] [rbp-130h]
  int v135; // [rsp+30h] [rbp-130h]
  __int64 v136; // [rsp+30h] [rbp-130h]
  __int64 v137; // [rsp+38h] [rbp-128h]
  char v138; // [rsp+40h] [rbp-120h]
  __int64 v139; // [rsp+40h] [rbp-120h]
  int v140; // [rsp+48h] [rbp-118h]
  _BOOL4 v141; // [rsp+4Ch] [rbp-114h]
  __m128i *v142; // [rsp+50h] [rbp-110h]
  __m128i *v143; // [rsp+50h] [rbp-110h]
  __m128i *v144; // [rsp+50h] [rbp-110h]
  __int64 v145; // [rsp+58h] [rbp-108h]
  __m128i *v146; // [rsp+58h] [rbp-108h]
  __int64 v147; // [rsp+58h] [rbp-108h]
  __int64 v148; // [rsp+58h] [rbp-108h]
  __int64 v149; // [rsp+58h] [rbp-108h]
  __int64 v150; // [rsp+58h] [rbp-108h]
  __int64 v151; // [rsp+60h] [rbp-100h]
  unsigned int v152; // [rsp+60h] [rbp-100h]
  __int64 v153; // [rsp+60h] [rbp-100h]
  __int64 v154; // [rsp+60h] [rbp-100h]
  __int64 v155; // [rsp+60h] [rbp-100h]
  __int64 v156; // [rsp+60h] [rbp-100h]
  __int64 v157; // [rsp+60h] [rbp-100h]
  __int64 v158; // [rsp+68h] [rbp-F8h]
  __int64 v159; // [rsp+68h] [rbp-F8h]
  int v160; // [rsp+68h] [rbp-F8h]
  __int64 v161; // [rsp+68h] [rbp-F8h]
  __int64 v162; // [rsp+68h] [rbp-F8h]
  int v163; // [rsp+68h] [rbp-F8h]
  __int64 v164; // [rsp+68h] [rbp-F8h]
  __int64 v165; // [rsp+68h] [rbp-F8h]
  __int64 v166; // [rsp+68h] [rbp-F8h]
  __int64 v167; // [rsp+68h] [rbp-F8h]
  __int64 v168; // [rsp+68h] [rbp-F8h]
  __int64 v169; // [rsp+68h] [rbp-F8h]
  int v170; // [rsp+70h] [rbp-F0h]
  char v172; // [rsp+78h] [rbp-E8h]
  char v173; // [rsp+7Dh] [rbp-E3h]
  char v174; // [rsp+7Fh] [rbp-E1h]
  int v175; // [rsp+80h] [rbp-E0h] BYREF
  int v176; // [rsp+84h] [rbp-DCh] BYREF
  __m128i *v177; // [rsp+88h] [rbp-D8h] BYREF
  __int64 *v178; // [rsp+90h] [rbp-D0h] BYREF
  const __m128i *v179; // [rsp+98h] [rbp-C8h] BYREF
  __m128i v180; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v181; // [rsp+B0h] [rbp-B0h]
  __m128i v182; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v183; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v184; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v185; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v186; // [rsp+100h] [rbp-60h] BYREF
  __m128i v187; // [rsp+110h] [rbp-50h] BYREF
  __int64 v188; // [rsp+120h] [rbp-40h]

  v140 = a4;
  switch ( a4 )
  {
    case 64:
      v140 = 0;
      a2 = sub_72C390();
      a3 = (const __m128i *)a2;
      v141 = dword_4D044AC != 0;
      goto LABEL_6;
    case 1024:
      v92 = (unsigned __int8 *)&unk_4F06A60;
LABEL_238:
      v141 = 0;
      a2 = (__int64)sub_72BA30(*v92);
      a3 = (const __m128i *)a2;
      v140 = 0;
      goto LABEL_6;
    case 2048:
      v92 = byte_4F06A51;
      goto LABEL_238;
  }
  v141 = 0;
  if ( a4 == 0x4000 )
  {
    v140 = 0;
    a2 = sub_72C570();
    a3 = (const __m128i *)a2;
  }
LABEL_6:
  v11 = *a1;
  if ( !a8 )
  {
LABEL_13:
    v117 = 0;
    v170 = 0;
LABEL_15:
    while ( *(_BYTE *)(v11 + 140) == 12 )
      v11 = *(_QWORD *)(v11 + 160);
    if ( (unsigned int)sub_8D23B0(v11) && (unsigned int)sub_8D3A70(v11) )
      sub_8AD220(v11, 0);
    v13 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)v11 + 96LL) + 40LL);
    v137 = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
    if ( v13 )
      v14 = v13[1];
    else
      v14 = 0;
    v172 = 0;
    v138 = 0;
    v173 = v170 ^ 1;
    while ( 1 )
    {
      v174 = (a2 != 0) & (v172 ^ 1);
      if ( !v14 )
        goto LABEL_58;
      while ( 1 )
      {
        v15 = *(_BYTE *)(v14 + 80);
        v178 = 0;
        v16 = v14;
        v181 = 0;
        v180 = 0;
        if ( v15 == 16 )
        {
          v17 = *(__int64 **)(v14 + 88);
          v16 = *v17;
          v15 = *(_BYTE *)(*v17 + 80);
        }
        if ( v15 == 24 )
        {
          v16 = *(_QWORD *)(v16 + 88);
          v15 = *(_BYTE *)(v16 + 80);
        }
        v18 = *(_QWORD *)(v16 + 88);
        if ( v15 == 20 )
        {
          if ( (*(_BYTE *)(v14 + 82) & 4) != 0 )
          {
            v34 = 0;
            v54 = 0;
            v35 = 0;
LABEL_99:
            sub_82B9D0(v14, 0, 0, 0, v54, v35, a11);
            goto LABEL_69;
          }
          v162 = *(_QWORD *)(v18 + 176);
          if ( (*(_BYTE *)(v162 + 194) & 1) != 0 && !v141 && a7 != 0 && (a10 & 0x800) == 0 )
          {
            v33 = 0;
            goto LABEL_54;
          }
          v153 = *(_QWORD *)(v162 + 152);
          v179 = a3;
          v177 = sub_73D790(v153);
          for ( i = v177; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
            ;
          if ( a5 || !v170 )
          {
            v144 = i;
            v65 = sub_8D5830(v179);
            v49 = v144;
            if ( !v65 )
              goto LABEL_88;
            if ( !a5 )
              goto LABEL_53;
            v66 = sub_73D7F0(v153);
            v67 = sub_8D32E0(v66);
            v49 = v144;
            if ( !v67 )
              goto LABEL_53;
            if ( v170 )
              goto LABEL_133;
LABEL_89:
            v50 = v49[8].m128i_i8[12];
            if ( v50 == 8 )
            {
              v177 = (__m128i *)sub_8D67C0(v177);
            }
            else if ( v50 == 7 )
            {
              v177 = (__m128i *)sub_6EE750((__int64)v177, 0);
            }
            if ( dword_4F077BC
              && (v79 = sub_73D7F0(v153), (unsigned int)sub_8D2FB0(v79))
              && (unsigned int)sub_8DBE70(v177) )
            {
              v51 = v177;
              j = **(_QWORD **)(v18 + 328);
            }
            else
            {
              v51 = v177;
              for ( j = **(_QWORD **)(v18 + 328); v51[8].m128i_i8[12] == 12; v51 = (__m128i *)v51[10].m128i_i64[0] )
                ;
            }
            for ( k = v179; k[8].m128i_i8[12] == 12; k = (const __m128i *)k[10].m128i_i64[0] )
              ;
          }
          else
          {
            v142 = i;
            v46 = sub_8D3A70(v179);
            v47 = v179;
            v48 = v142;
            if ( v46 )
            {
              v77 = sub_73D7F0(v153);
              v78 = sub_8D32E0(v77);
              v47 = v179;
              v48 = v142;
              if ( !v78 )
              {
                while ( v47[8].m128i_i8[12] == 12 )
                  v47 = (const __m128i *)v47[10].m128i_i64[0];
                v179 = v47;
              }
            }
            v143 = v48;
            if ( (unsigned int)sub_8D5830(v47) )
              goto LABEL_53;
            v49 = v143;
LABEL_88:
            if ( !v170 )
              goto LABEL_89;
LABEL_133:
            v51 = v177;
            k = v179;
            j = **(_QWORD **)(v18 + 328);
          }
          if ( !(unsigned int)sub_8B3500(k, v51, &v178, j, 0)
            && !(unsigned int)sub_8B4FF0(v179, v177, &v178, **(_QWORD **)(v18 + 328), 68) )
          {
            if ( !v170 )
              goto LABEL_53;
            sub_828750(&v177, &v179);
            if ( !(unsigned int)sub_8B3500(v179, v177, &v178, **(_QWORD **)(v18 + 328), 0) )
              goto LABEL_53;
          }
          v68 = sub_8B2240(&v178, v16, 0, 0x20000, 0);
          if ( !v68 )
            goto LABEL_53;
          if ( dword_4D04494 )
          {
            v69 = *(unsigned int *)(v18 + 392);
            if ( v69 > unk_4D042F0 )
            {
              sub_861C90();
LABEL_53:
              v33 = v178;
LABEL_54:
              sub_725130(v33);
              goto LABEL_55;
            }
            v70 = v178;
            *(_DWORD *)(v18 + 392) = v69 + 1;
            v71 = *(_BYTE *)(v16 + 80);
            if ( v71 == 16 )
            {
              v16 = **(_QWORD **)(v16 + 88);
              v71 = *(_BYTE *)(v16 + 80);
            }
            if ( v71 == 24 )
            {
              v16 = *(_QWORD *)(v16 + 88);
              v71 = *(_BYTE *)(v16 + 80);
            }
            if ( (unsigned __int8)(v71 - 10) <= 1u )
            {
              v87 = *(_QWORD *)(v16 + 88);
              if ( (*(_BYTE *)(v87 + 194) & 0x40) != 0 )
              {
                do
                  v87 = *(_QWORD *)(v87 + 232);
                while ( (*(_BYTE *)(v87 + 194) & 0x40) != 0 );
                goto LABEL_220;
              }
            }
            else if ( v71 == 20 )
            {
              v95 = *(_QWORD *)(*(_QWORD *)(v16 + 88) + 176LL);
              if ( (*(_BYTE *)(v95 + 194) & 0x40) != 0 )
              {
                do
                  v95 = *(_QWORD *)(v95 + 232);
                while ( (*(_BYTE *)(v95 + 194) & 0x40) != 0 );
                v87 = *(_QWORD *)(v95 + 248);
LABEL_220:
                v16 = *(_QWORD *)v87;
              }
            }
            v154 = v68;
            v72 = sub_8A00C0(v16, v70, 0);
            v68 = v154;
            if ( !v72 )
            {
              --*(_DWORD *)(v18 + 392);
              goto LABEL_53;
            }
            --*(_DWORD *)(v18 + 392);
          }
          if ( a7 )
          {
            if ( !(v141 | a10 & 0x400000) && *(char *)(*(_QWORD *)(v68 + 168) + 20LL) < 0 )
            {
              v97 = *(__int64 **)(v68 + 104);
              v157 = v68;
              v98 = sub_736C60(84, v97);
              v68 = v157;
              v102 = *(_QWORD *)(v98[4] + 40);
              if ( *(_BYTE *)(v102 + 173) != 12 )
              {
                if ( !(unsigned int)sub_711520(v102, (__int64)v97, v99, v100, v101) )
                  goto LABEL_53;
                v68 = v157;
              }
            }
          }
          v155 = v68;
          if ( !sub_5F1C40(v68) )
            goto LABEL_53;
          v19 = v155;
          v18 = v162;
          goto LABEL_32;
        }
        if ( (*(_BYTE *)(v14 + 82) & 4) != 0 )
        {
          v34 = 0;
          v35 = 0;
          goto LABEL_68;
        }
        if ( !v141 && a7 != 0 && (*(_BYTE *)(v18 + 194) & 1) != 0 && (a10 & 0x800) == 0 )
        {
          v33 = 0;
          goto LABEL_54;
        }
        v19 = *(_QWORD *)(v18 + 152);
        if ( (*(_BYTE *)(v18 + 207) & 0x30) == 0x10 )
        {
          v165 = *(_QWORD *)(v18 + 152);
          sub_8B1A30(*(_QWORD *)(v16 + 88), dword_4F07508);
          v19 = v165;
        }
LABEL_32:
        while ( *(_BYTE *)(v19 + 140) == 12 )
          v19 = *(_QWORD *)(v19 + 160);
        v158 = v19;
        v20 = sub_73D790(v19);
        v21 = v20[8].m128i_i8[12] == 12;
        v177 = v20;
        v22 = v20;
        if ( v21 )
        {
          do
            v22 = (const __m128i *)v22[10].m128i_i64[0];
          while ( v22[8].m128i_i8[12] == 12 );
        }
        v151 = v158;
        v159 = *(_QWORD *)(v158 + 160);
        v23 = sub_8D32E0(v159);
        v24 = v151;
        v25 = v23;
        if ( v23 )
        {
          v122 = v23;
          v42 = v159;
          v163 = sub_8D30C0(v159);
          v43 = sub_8D3110(v42);
          v24 = v151;
          v25 = v122;
          v152 = v43;
          if ( v43 )
          {
            v156 = v24;
            v73 = sub_8D3190();
            v24 = v156;
            v25 = v122;
            if ( v73 && v22[8].m128i_i8[12] == 7 )
            {
              v152 = 0;
              if ( v117 )
              {
                v27 = 0;
LABEL_76:
                v123 = v27;
                v127 = v25;
                v164 = v24;
                v44 = sub_8D3190();
                v24 = v164;
                v25 = v127;
                if ( !v44 || v22[8].m128i_i8[12] != 7 || v123 && a9 != 0 && (v152 & 1) == 0 )
                  goto LABEL_53;
              }
              v160 = 1;
              goto LABEL_40;
            }
            v152 = 1;
          }
          v27 = v163 == 0;
          if ( a5 && !v163 )
            goto LABEL_53;
          v160 = v163 != 0;
          if ( (v160 & v117) != 0 )
            goto LABEL_76;
          v26 = (v152 ^ 1) & 1;
        }
        else
        {
          if ( a5 )
            goto LABEL_53;
          v152 = 0;
          v26 = 1;
          v27 = 1;
          v160 = 0;
        }
        if ( a9 && ((unsigned __int8)v26 & v27) != 0 && (unsigned __int8)(v22[8].m128i_i8[12] - 9) > 2u )
          goto LABEL_53;
LABEL_40:
        v28 = v177;
        if ( !v140 && a2 )
        {
          v121 = v25;
          v126 = v24;
          v29 = sub_8DED30(v177, a2, 19);
          v30 = v126;
          v31 = v22[8].m128i_i8[12];
          v32 = v121;
          if ( (unsigned __int8)(v31 - 9) > 2u )
          {
            if ( !v121 || a9 )
            {
LABEL_49:
              if ( !v29 )
                goto LABEL_50;
            }
            else if ( v173 || !v29 )
            {
              if ( v31 == 8 )
              {
                v113 = (__m128i *)sub_8D67C0(v177);
                v108 = v126;
                v177 = v113;
                v22 = v113;
              }
              else
              {
                if ( v31 != 7 )
                  goto LABEL_49;
                v107 = (__m128i *)sub_6EE750((__int64)v177, 0);
                v108 = v126;
                v177 = v107;
                v22 = v107;
              }
              v169 = v108;
              v109 = sub_8DED30(v22, a2, 19);
              v30 = v169;
              if ( v109 )
              {
LABEL_258:
                LOBYTE(v160) = 0;
                goto LABEL_204;
              }
LABEL_50:
              if ( v170
                || (v166 = v30,
                    v88 = sub_8E1010((_DWORD)v177, 0, 0, 0, a7, 0, a2, 0, 0, 1, 0, (__int64)&v180, 0),
                    v30 = v166,
                    !v88) )
              {
                if ( (a10 & 0x100) == 0 )
                  goto LABEL_53;
                v161 = v30;
                if ( !(unsigned int)sub_8D2E30(v22) )
                  goto LABEL_53;
                if ( !(unsigned int)sub_8D2E30(a2) )
                  goto LABEL_53;
                v150 = v161;
                v168 = sub_8D46C0(a2);
                v103 = sub_8D46C0(v22);
                v106 = sub_8D97D0(v103, v168, 32, v104, v105);
                v30 = v150;
                if ( !v106 )
                  goto LABEL_53;
                goto LABEL_258;
              }
              if ( *(_QWORD *)(v18 + 240) && (v180.m128i_i8[12] & 0x20) != 0 )
                goto LABEL_53;
              LOBYTE(v160) = 0;
              if ( (a10 & 0x80000) != 0 )
              {
                v167 = v30;
                v179 = (const __m128i *)sub_724DC0();
                if ( !(unsigned int)sub_6E47F0(v18, (__int64)a1, (__int64)v177, (__int64)v179) )
                {
                  sub_724E30((__int64)&v179);
                  goto LABEL_53;
                }
                v111 = sub_8DD4B0(v177, 1, v179, a2, 0);
                sub_724E30((__int64)&v179);
                v30 = v167;
                if ( !v111 )
                  goto LABEL_53;
                goto LABEL_258;
              }
LABEL_204:
              if ( (*(_BYTE *)(v18 + 194) & 1) == 0 )
                goto LABEL_115;
              goto LABEL_205;
            }
            if ( v160 )
            {
              if ( (v177[8].m128i_i8[12] & 0xFB) != 8 || (v112 = sub_8D5780(a2, v177), v30 = v126, !v112) )
              {
                if ( (*(_BYTE *)(v18 + 194) & 1) == 0 || (v180.m128i_i8[12] & 0x20) == 0 )
                  goto LABEL_117;
LABEL_206:
                if ( !dword_4F077BC || qword_4F077A8 > 0x9EFBu )
                  goto LABEL_53;
                goto LABEL_115;
              }
              LOBYTE(v160) = a9 | v117;
              if ( a9 | v117 || (unsigned __int8)(v22[8].m128i_i8[12] - 7) <= 1u )
                goto LABEL_53;
            }
            goto LABEL_204;
          }
          if ( v29 )
          {
            v148 = 0;
          }
          else
          {
            if ( !(a6 | v170) )
              goto LABEL_53;
            if ( !(unsigned int)sub_8D3A70(a2) )
              goto LABEL_53;
            v74 = sub_8D5CE0(v22, a2);
            v30 = v126;
            v32 = v121;
            v148 = v74;
            if ( !v74 )
              goto LABEL_53;
          }
          if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
          {
            v115 = v32;
            v125 = v30;
            v89 = sub_8D4C10(a2, dword_4F077C4 != 2);
            v80 = v177;
            v30 = v125;
            v32 = v115;
            v81 = v89 & 0xFFFFFF8F;
            v83 = 0;
            if ( (v177[8].m128i_i8[12] & 0xFB) == 8 )
            {
LABEL_190:
              v114 = v81;
              v124 = v32;
              v131 = v30;
              v82 = sub_8D4C10(v80, dword_4F077C4 != 2);
              v81 = v114;
              v32 = v124;
              v30 = v131;
              v83 = v82 & 0xFFFFFF8F;
            }
            if ( v83 != v81 )
            {
              if ( a8 || v170 )
              {
                if ( (v177[8].m128i_i8[12] & 0xFB) == 8 )
                {
                  v135 = v32;
                  v139 = v30;
                  v94 = sub_8D5780(a2, v177);
                  v30 = v139;
                  v32 = v135;
                  if ( v94 )
                  {
LABEL_252:
                    v138 = 1;
                    v33 = v178;
                    goto LABEL_54;
                  }
                }
              }
              else if ( !a6 && qword_4D0495C )
              {
                goto LABEL_252;
              }
              v138 = 1;
            }
          }
          else
          {
            v80 = v177;
            if ( (v177[8].m128i_i8[12] & 0xFB) == 8 )
            {
              v81 = 0;
              goto LABEL_190;
            }
          }
          if ( v32 && v173 && (*(_BYTE *)(*(_QWORD *)(v22->m128i_i64[0] + 96) + 177LL) & 1) != 0 )
          {
            v84 = 0;
            v85 = (__int64)a1 + 68;
            if ( (v177[8].m128i_i8[12] & 0xFB) == 8 )
            {
              v136 = v30;
              v110 = sub_8D4C10(v177, dword_4F077C4 != 2);
              v85 = (__int64)a1 + 68;
              v30 = v136;
              v84 = v110;
            }
            v132 = v30;
            v86 = sub_83DE00(v22, v84, v152, 0, v85, &v175, &v176, 0, &v179);
            if ( !v86 || (*(_BYTE *)(*(_QWORD *)(v86 + 88) + 206LL) & 0x10) != 0 )
              goto LABEL_53;
            v30 = v132;
            if ( !v148 )
              goto LABEL_204;
          }
          else if ( !v148 )
          {
            goto LABEL_204;
          }
          v180.m128i_i8[12] |= 0x20u;
          v180.m128i_i64[0] = v148;
          if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
          {
            v149 = v30;
            v96 = sub_8D5780(v177, a2);
            v30 = v149;
            v91 = v96 & 1;
          }
          else
          {
            v91 = 0;
          }
          v138 = 1;
          v180.m128i_i8[12] = v180.m128i_i8[12] & 0xFD | (2 * v91);
          if ( (*(_BYTE *)(v18 + 194) & 1) == 0 )
            goto LABEL_115;
LABEL_205:
          if ( (v180.m128i_i8[12] & 0x20) != 0 )
            goto LABEL_206;
          goto LABEL_115;
        }
        if ( v25 && v173 )
        {
          v55 = v22[8].m128i_i8[12];
          if ( v55 == 8 )
          {
            v133 = v24;
            v90 = (__m128i *)sub_8D67C0(v177);
            v24 = v133;
            LOBYTE(v160) = 0;
            v177 = v90;
            v28 = v90;
          }
          else if ( v55 == 7 )
          {
            v134 = v24;
            v93 = (__m128i *)sub_6EE750((__int64)v177, 0);
            v24 = v134;
            LOBYTE(v160) = 0;
            v177 = v93;
            v28 = v93;
          }
        }
        if ( a5 && (v28[8].m128i_i8[12] & 0xFB) == 8 )
        {
          v130 = v24;
          v76 = sub_8D4C10(v28, dword_4F077C4 != 2);
          v24 = v130;
          if ( (v76 & 1) != 0 )
            goto LABEL_53;
          v28 = v177;
        }
        v128 = v24;
        v56 = sub_831460((__int64)v28, v140);
        v30 = v128;
        if ( v56 )
        {
          if ( !a2 || (a10 & 0x80000) == 0 )
            goto LABEL_115;
        }
        else
        {
          if ( (v140 & 1) == 0 )
            goto LABEL_53;
          v57 = sub_8D28F0(v177);
          v30 = v128;
          if ( !v57 )
            goto LABEL_53;
          v180.m128i_i8[12] |= 0x20u;
          if ( !a2 )
            goto LABEL_115;
          v58 = sub_8D6540(v177);
          v30 = v128;
          if ( a2 == v58 || (v59 = sub_8DED30(v58, a2, 3), v30 = v128, v59) )
            v180.m128i_i8[12] |= 0x40u;
          if ( (a10 & 0x80000) == 0 )
            goto LABEL_115;
        }
        v129 = v30;
        v75 = sub_8E1010((_DWORD)v177, 0, 0, 0, a7, 0, a2, 0, 0, 1, 0, (__int64)&v180, 0);
        v30 = v129;
        if ( !v75 )
          goto LABEL_53;
LABEL_115:
        if ( (v160 & 1) == 0 && a5 )
          goto LABEL_53;
LABEL_117:
        v60 = *(_QWORD *)v18;
        if ( (*(_BYTE *)(*(_QWORD *)v18 + 104LL) & 1) != 0 )
        {
          v147 = v30;
          v62 = sub_8796F0(v60);
          v30 = v147;
        }
        else
        {
          v61 = *(_QWORD *)(v60 + 88);
          if ( *(_BYTE *)(v60 + 80) == 20 )
            v61 = *(_QWORD *)(v61 + 176);
          v62 = (*(_BYTE *)(v61 + 208) & 4) != 0;
        }
        if ( v62 )
          goto LABEL_53;
        v145 = v30;
        v63 = sub_82EAF0(v30, v14, 1);
        if ( v63 )
          sub_839CB0((__int64)a1, v18, v145, v63, (__int64)&v182);
        else
          sub_838020(
            (__int64)a1,
            0,
            *(__m128i **)(**(_QWORD **)(v145 + 168) + 8LL),
            **(_QWORD **)(v145 + 168),
            0,
            0,
            &v182);
        if ( v182.m128i_i32[2] == 7 )
          goto LABEL_53;
        v64 = (__m128i *)qword_4D03C60;
        if ( qword_4D03C60 )
          qword_4D03C60 = (_QWORD *)*qword_4D03C60;
        else
          v64 = (__m128i *)sub_823970(104);
        v146 = v64;
        sub_82D850((__int64)v64);
        v35 = (__int64)v146;
        *v146 = _mm_loadu_si128(&v182);
        v34 = (v152 | v160) & 1;
        v54 = (__int64)v178;
        v146[1] = _mm_loadu_si128(&v183);
        v146[2] = _mm_loadu_si128(&v184);
        v146[3] = _mm_loadu_si128(&v185);
        v146[4] = _mm_loadu_si128(&v186);
        v146[5] = _mm_loadu_si128(&v187);
        v146[6].m128i_i64[0] = v188;
        if ( v15 == 20 )
          goto LABEL_99;
LABEL_68:
        sub_82B8E0(v14, 0, v35, a11);
        v36 = *a11;
        *(_QWORD *)(v36 + 64) = v18;
        *(_QWORD *)(v36 + 72) = v14;
LABEL_69:
        v37 = *a11;
        v38 = *(_BYTE *)(*a11 + 80);
        *(_BYTE *)(v37 + 145) |= 0x40u;
        *(_BYTE *)(v37 + 80) = (16 * v138) | v38 & 0xEF;
        if ( v170 )
          v180.m128i_i8[12] = v180.m128i_i8[12] & 0xF9 | (2 * v180.m128i_i8[12]) & 4;
        v39 = _mm_loadu_si128(&v180);
        *(_QWORD *)(v37 + 104) = v181;
        v40 = 4 * v34;
        v41 = *(_BYTE *)(v37 + 80);
        *(__m128i *)(v37 + 88) = v39;
        *(_BYTE *)(v37 + 80) = v40 | v41 & 0xFB;
LABEL_55:
        if ( v13 )
        {
          v13 = (_QWORD *)*v13;
          if ( v13 )
            break;
        }
LABEL_58:
        if ( !v174 )
          return;
        v13 = *(_QWORD **)(v137 + 48);
        if ( !v13 )
          return;
        v14 = v13[1];
        if ( !v14 )
          return;
        v174 = 0;
        v172 = 1;
      }
      v14 = v13[1];
    }
  }
  v117 = sub_8D3110(a8);
  if ( !v117 )
  {
    if ( a9 )
    {
      v170 = 1;
      if ( !a5 )
        a5 = sub_8D4D20(a8) == 0;
      goto LABEL_15;
    }
    goto LABEL_13;
  }
  if ( !a9 )
  {
    v170 = 0;
    v117 = 1;
    goto LABEL_15;
  }
  v12 = sub_8D4D20(a8);
  if ( !a5 || !v12 )
  {
    v117 = 1;
    v170 = 1;
    a5 = v12 == 0;
    goto LABEL_15;
  }
}
