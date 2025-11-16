// Function: sub_265EF70
// Address: 0x265ef70
//
void __fastcall sub_265EF70(__int64 a1, __int64 a2)
{
  __m128i *v2; // rax
  unsigned __int64 v3; // rcx
  __m128i *v4; // rax
  __int64 v5; // rcx
  __int64 *v6; // rdi
  __int64 v7; // r8
  bool v8; // al
  bool v9; // al
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char v22; // al
  const char *v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  signed __int64 v34; // r15
  __int64 v35; // rdx
  __int64 **v36; // rbx
  __int64 **v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r8
  char v42; // al
  void *v43; // rax
  const char *v44; // rsi
  __int64 v45; // rax
  __int32 v46; // eax
  __int64 v47; // rdx
  __int64 v48; // r12
  unsigned __int32 v49; // ebx
  __int64 (__fastcall **v50)(); // rax
  void *v51; // rax
  __int64 v52; // rax
  void *v53; // rax
  __int64 v54; // rax
  int v55; // r8d
  int v56; // r8d
  void *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned __int64 v63; // r15
  char v64; // r15
  __m128i *v65; // rsi
  __int64 v66; // r14
  __int64 v67; // rax
  __m128i *p_src; // rsi
  __int64 v69; // r14
  __int64 v70; // rax
  void *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // esi
  _DWORD *v76; // r9
  int v77; // edi
  __int64 v78; // r15
  unsigned int v79; // edx
  _DWORD *v80; // r8
  int v81; // edi
  int v82; // r9d
  __int64 v83; // rbx
  char v84; // bl
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // r14
  __int64 v88; // r14
  __int64 v89; // rax
  char v90; // al
  unsigned __int32 v91; // ebx
  __int64 v92; // r14
  unsigned int v93; // edx
  int *v94; // r15
  int v95; // ecx
  int v96; // r8d
  __int64 v97; // rax
  __int64 v98; // rax
  int v99; // ecx
  int v100; // ecx
  int v101; // edi
  char v102; // [rsp+1Fh] [rbp-421h]
  unsigned int v103; // [rsp+28h] [rbp-418h]
  unsigned __int64 *v104; // [rsp+38h] [rbp-408h]
  unsigned __int64 v105; // [rsp+48h] [rbp-3F8h]
  __int64 v106; // [rsp+48h] [rbp-3F8h]
  __int64 v107; // [rsp+48h] [rbp-3F8h]
  __int64 v108; // [rsp+50h] [rbp-3F0h]
  __int64 v109; // [rsp+50h] [rbp-3F0h]
  char v110; // [rsp+50h] [rbp-3F0h]
  __int64 v111; // [rsp+50h] [rbp-3F0h]
  __int64 v112; // [rsp+50h] [rbp-3F0h]
  _DWORD *v113; // [rsp+50h] [rbp-3F0h]
  _DWORD *v114; // [rsp+50h] [rbp-3F0h]
  unsigned __int64 *v116; // [rsp+70h] [rbp-3D0h]
  unsigned __int64 v117; // [rsp+78h] [rbp-3C8h]
  unsigned int v118; // [rsp+8Ch] [rbp-3B4h] BYREF
  __int64 v119[2]; // [rsp+90h] [rbp-3B0h] BYREF
  _QWORD v120[2]; // [rsp+A0h] [rbp-3A0h] BYREF
  unsigned __int64 v121[2]; // [rsp+B0h] [rbp-390h] BYREF
  __m128i v122; // [rsp+C0h] [rbp-380h] BYREF
  __m128i dest; // [rsp+D0h] [rbp-370h] BYREF
  __m128i v124; // [rsp+E0h] [rbp-360h] BYREF
  __int64 v125; // [rsp+F0h] [rbp-350h] BYREF
  __int64 v126; // [rsp+F8h] [rbp-348h]
  unsigned __int8 *v127; // [rsp+110h] [rbp-330h] BYREF
  size_t v128; // [rsp+118h] [rbp-328h]
  _BYTE v129[32]; // [rsp+130h] [rbp-310h] BYREF
  unsigned __int64 v130[4]; // [rsp+150h] [rbp-2F0h] BYREF
  unsigned __int64 v131[4]; // [rsp+170h] [rbp-2D0h] BYREF
  void *v132; // [rsp+190h] [rbp-2B0h] BYREF
  __int16 v133; // [rsp+1B0h] [rbp-290h]
  void *v134; // [rsp+1C0h] [rbp-280h] BYREF
  __int16 v135; // [rsp+1E0h] [rbp-260h]
  __m128i v136[2]; // [rsp+1F0h] [rbp-250h] BYREF
  char v137; // [rsp+210h] [rbp-230h]
  char v138; // [rsp+211h] [rbp-22Fh]
  __m128i v139[2]; // [rsp+220h] [rbp-220h] BYREF
  __int16 v140; // [rsp+240h] [rbp-200h]
  __m128i v141[3]; // [rsp+250h] [rbp-1F0h] BYREF
  __m128i v142[2]; // [rsp+280h] [rbp-1C0h] BYREF
  char v143; // [rsp+2A0h] [rbp-1A0h]
  char v144; // [rsp+2A1h] [rbp-19Fh]
  __m128i v145[2]; // [rsp+2B0h] [rbp-190h] BYREF
  char v146; // [rsp+2D0h] [rbp-170h]
  char v147; // [rsp+2D1h] [rbp-16Fh]
  __m128i v148[2]; // [rsp+2E0h] [rbp-160h] BYREF
  __int16 v149; // [rsp+300h] [rbp-140h]
  __m128i v150; // [rsp+310h] [rbp-130h] BYREF
  char v151; // [rsp+320h] [rbp-120h] BYREF
  __m128i src; // [rsp+340h] [rbp-100h] BYREF
  __int64 v153; // [rsp+350h] [rbp-F0h] BYREF
  unsigned int v154; // [rsp+358h] [rbp-E8h]
  char v155; // [rsp+360h] [rbp-E0h]
  char v156; // [rsp+361h] [rbp-DFh]
  __m128i v157; // [rsp+370h] [rbp-D0h] BYREF
  __m128i v158; // [rsp+380h] [rbp-C0h] BYREF
  __int64 v159; // [rsp+390h] [rbp-B0h]
  __int64 v160; // [rsp+398h] [rbp-A8h]
  __m128i *v161; // [rsp+3A0h] [rbp-A0h]
  __m128i *p_dest; // [rsp+3B0h] [rbp-90h] BYREF
  __int64 (__fastcall **v163)(); // [rsp+3B8h] [rbp-88h]
  __int16 v164; // [rsp+3D0h] [rbp-70h]

  v119[0] = (__int64)v120;
  sub_2640340(v119, (_BYTE *)qword_4FF4168, qword_4FF4168 + qword_4FF4170);
  sub_2241520((unsigned __int64 *)v119, "ccg.");
  v2 = (__m128i *)sub_2241490((unsigned __int64 *)v119, *(char **)a2, *(_QWORD *)(a2 + 8));
  v121[0] = (unsigned __int64)&v122;
  if ( (__m128i *)v2->m128i_i64[0] == &v2[1] )
  {
    v122 = _mm_loadu_si128(v2 + 1);
  }
  else
  {
    v121[0] = v2->m128i_i64[0];
    v122.m128i_i64[0] = v2[1].m128i_i64[0];
  }
  v3 = v2->m128i_u64[1];
  v2[1].m128i_i8[0] = 0;
  v121[1] = v3;
  v2->m128i_i64[0] = (__int64)v2[1].m128i_i64;
  v2->m128i_i64[1] = 0;
  v4 = (__m128i *)sub_2241520(v121, ".dot");
  dest.m128i_i64[0] = (__int64)&v124;
  if ( (__m128i *)v4->m128i_i64[0] == &v4[1] )
  {
    v124 = _mm_loadu_si128(v4 + 1);
  }
  else
  {
    dest.m128i_i64[0] = v4->m128i_i64[0];
    v124.m128i_i64[0] = v4[1].m128i_i64[0];
  }
  v5 = v4->m128i_i64[1];
  v4[1].m128i_i8[0] = 0;
  dest.m128i_i64[1] = v5;
  v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
  v4->m128i_i64[1] = 0;
  v134 = (void *)a2;
  v135 = 260;
  v133 = 257;
  if ( !dest.m128i_i64[1] )
  {
    sub_CA0F50(v157.m128i_i64, &v132);
    v164 = 260;
    p_dest = &v157;
    sub_C67360(src.m128i_i64, (__int64)&p_dest, &v118);
    v6 = (__int64 *)dest.m128i_i64[0];
    if ( (__int64 *)src.m128i_i64[0] == &v153 )
    {
      v74 = src.m128i_i64[1];
      if ( src.m128i_i64[1] )
      {
        if ( src.m128i_i64[1] == 1 )
          *(_BYTE *)dest.m128i_i64[0] = v153;
        else
          memcpy((void *)dest.m128i_i64[0], (const void *)src.m128i_i64[0], src.m128i_u64[1]);
        v74 = src.m128i_i64[1];
        v6 = (__int64 *)dest.m128i_i64[0];
      }
      dest.m128i_i64[1] = v74;
      *((_BYTE *)v6 + v74) = 0;
      v6 = (__int64 *)src.m128i_i64[0];
      goto LABEL_10;
    }
    if ( (__m128i *)dest.m128i_i64[0] == &v124 )
    {
      dest = src;
      v124.m128i_i64[0] = v153;
    }
    else
    {
      v7 = v124.m128i_i64[0];
      dest = src;
      v124.m128i_i64[0] = v153;
      if ( v6 )
      {
        src.m128i_i64[0] = (__int64)v6;
        v153 = v7;
LABEL_10:
        src.m128i_i64[1] = 0;
        *(_BYTE *)v6 = 0;
        sub_2240A30((unsigned __int64 *)&src);
        sub_2240A30((unsigned __int64 *)&v157);
        goto LABEL_11;
      }
    }
    src.m128i_i64[0] = (__int64)&v153;
    v6 = &v153;
    goto LABEL_10;
  }
  v164 = 260;
  p_dest = &dest;
  v46 = sub_C83360((__int64)&p_dest, (int *)&v118, 0, 2, 1, 0x1B6u);
  v48 = v47;
  src.m128i_i64[1] = v47;
  v49 = v46;
  src.m128i_i32[0] = v46;
  v50 = sub_2241E50();
  LODWORD(p_dest) = 17;
  v163 = v50;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __m128i **))(*(_QWORD *)v48 + 48LL))(v48, v49, &p_dest)
    || (*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), __m128i *, _QWORD))*v163 + 7))(
         v163,
         &src,
         (unsigned int)p_dest) )
  {
    v51 = sub_CB72A0();
    v52 = sub_904010((__int64)v51, "file exists, overwriting");
    sub_904010(v52, "\n");
  }
  else
  {
    if ( src.m128i_i32[0] )
    {
      v53 = sub_CB72A0();
      v54 = sub_904010((__int64)v53, "error writing into file");
      sub_904010(v54, "\n");
      sub_263F570(v157.m128i_i64, byte_3F871B3);
      goto LABEL_73;
    }
    v71 = sub_CB72A0();
    v72 = sub_904010((__int64)v71, "writing to the newly created file ");
    v73 = sub_CB6200(v72, (unsigned __int8 *)dest.m128i_i64[0], dest.m128i_u64[1]);
    sub_904010(v73, "\n");
  }
LABEL_11:
  sub_CB6EE0((__int64)&p_dest, v118, 1, 0, 0);
  if ( v118 != -1 )
  {
    if ( !(unsigned int)sub_23DF0D0(&dword_4FF3CC8) || (v8 = 1, dword_4FF3E28) )
    {
      v56 = sub_23DF0D0(&dword_4FF3BE8);
      v8 = 0;
      if ( v56 )
        v8 = dword_4FF3E28 != 2;
    }
    byte_4FF31E8 = v8;
    if ( !(unsigned int)sub_23DF0D0(&dword_4FF3CC8) || (v9 = 1, dword_4FF3E28) )
    {
      v55 = sub_23DF0D0(&dword_4FF3BE8);
      v9 = 0;
      if ( v55 )
        v9 = dword_4FF3E28 != 2;
    }
    byte_4FF31E8 = v9;
    sub_CA0F50(&v125, &v134);
    sub_263F570(src.m128i_i64, byte_3F871B3);
    if ( v126 )
    {
      v65 = (__m128i *)&v125;
      v66 = sub_904010((__int64)&p_dest, "digraph \"");
    }
    else
    {
      if ( !src.m128i_i64[1] )
      {
        sub_904010((__int64)&p_dest, "digraph unnamed {\n");
        if ( !v126 )
        {
LABEL_19:
          if ( !src.m128i_i64[1] )
            goto LABEL_20;
          p_src = &src;
          v69 = sub_904010((__int64)&p_dest, "\tlabel=\"");
LABEL_99:
          sub_C67200(v157.m128i_i64, (__int64)p_src);
          v70 = sub_CB6200(v69, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
          sub_904010(v70, "\";\n");
          sub_2240A30((unsigned __int64 *)&v157);
LABEL_20:
          sub_263F570(v157.m128i_i64, byte_3F871B3);
          sub_CB6200((__int64)&p_dest, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
          sub_2240A30((unsigned __int64 *)&v157);
          sub_904010((__int64)&p_dest, "\n");
          sub_2240A30((unsigned __int64 *)&src);
          v104 = *(unsigned __int64 **)(a1 + 328);
          v116 = *(unsigned __int64 **)(a1 + 320);
          if ( v104 == v116 )
          {
LABEL_57:
            sub_904010((__int64)&p_dest, "}\n");
            sub_2240A30((unsigned __int64 *)&v125);
            v43 = sub_CB72A0();
            v44 = " done. \n";
            sub_904010((__int64)v43, " done. \n");
            v157.m128i_i64[0] = (__int64)&v158;
            if ( (__m128i *)dest.m128i_i64[0] == &v124 )
            {
              v158 = _mm_load_si128(&v124);
            }
            else
            {
              v157.m128i_i64[0] = dest.m128i_i64[0];
              v158.m128i_i64[0] = v124.m128i_i64[0];
            }
            v45 = dest.m128i_i64[1];
            v124.m128i_i8[0] = 0;
            dest.m128i_i64[1] = 0;
            v157.m128i_i64[1] = v45;
            dest.m128i_i64[0] = (__int64)&v124;
            goto LABEL_72;
          }
          while ( 1 )
          {
            v117 = *v116;
            if ( *(_BYTE *)(*v116 + 2) )
              break;
LABEL_56:
            if ( v104 == ++v116 )
              goto LABEL_57;
          }
          if ( dword_4FF3E28 != 1 )
          {
            if ( dword_4FF3E28 != 2 )
              goto LABEL_24;
            sub_264D230((__int64)&v157, v117, v10);
            v91 = v158.m128i_u32[2];
            v92 = v157.m128i_i64[1];
            if ( v158.m128i_i32[2] )
            {
              v93 = (v158.m128i_i32[2] - 1) & (37 * qword_4FF3C68);
              v94 = (int *)(v157.m128i_i64[1] + 4LL * v93);
              v95 = *v94;
              if ( (_DWORD)qword_4FF3C68 == *v94 )
              {
LABEL_133:
                sub_2342640((__int64)&v157);
                if ( v94 == (int *)(v92 + 4LL * v91) )
                  goto LABEL_56;
LABEL_24:
                sub_264D230((__int64)v129, v117, v10);
                v11 = byte_4FF31E8;
                if ( byte_4FF31E8 )
                {
                  if ( (unsigned int)sub_23DF0D0(&dword_4FF3BE8) )
                  {
                    v157.m128i_i32[0] = qword_4FF3C68;
                    v11 = sub_264A710((__int64)v129, v157.m128i_i32);
                  }
                  else
                  {
                    v11 = sub_265E7D0((__int64)v129, a1 + 96);
                  }
                }
                v102 = v11;
                src.m128i_i64[0] = (__int64)"\"";
                v156 = 1;
                v155 = 3;
                sub_26446F0((__int64 *)v131, (__int64)v129);
                v149 = 260;
                v148[0].m128i_i64[0] = (__int64)v131;
                v142[0].m128i_i64[0] = (__int64)" ";
                v144 = 1;
                v143 = 3;
                sub_2640980((__int64)v130, v117);
                v140 = 260;
                v139[0].m128i_i64[0] = (__int64)v130;
                v136[0].m128i_i64[0] = (__int64)"tooltip=\"";
                v138 = 1;
                v137 = 3;
                sub_9C6370(v141, v136, v139, 260, v12, v13);
                sub_9C6370(v145, v141, v142, v14, v15, v16);
                sub_9C6370(&v150, v145, v148, (__int64)v148, v17, v18);
                sub_9C6370(&v157, &v150, &src, v19, v20, v21);
                sub_CA0F50((__int64 *)&v127, (void **)&v157);
                sub_2240A30(v130);
                sub_2240A30(v131);
                if ( v102 )
                {
                  sub_2241520((unsigned __int64 *)&v127, ",fontsize=\"30\"");
                  v156 = 1;
                  src.m128i_i64[0] = (__int64)"\"";
                  v155 = 3;
                  v90 = *(_BYTE *)(v117 + 2);
                  if ( v90 != 1 )
                  {
                    if ( v90 == 2 )
                    {
                      v23 = "cyan";
                    }
                    else
                    {
                      v23 = "magenta";
                      if ( v90 != 3 )
                        goto LABEL_29;
                    }
                    goto LABEL_30;
                  }
                }
                else
                {
                  v156 = 1;
                  src.m128i_i64[0] = (__int64)"\"";
                  v155 = 3;
                  v22 = *(_BYTE *)(v117 + 2);
                  if ( v22 != 1 )
                  {
                    if ( v22 == 2 )
                    {
                      v23 = "cyan";
                      if ( byte_4FF31E8 )
                        v23 = "lightskyblue";
                    }
                    else
                    {
                      if ( v22 != 3 )
                      {
LABEL_29:
                        v23 = "gray";
                        goto LABEL_30;
                      }
                      v23 = "mediumorchid1";
                    }
LABEL_30:
                    sub_263F570(v141[0].m128i_i64, v23);
                    v148[0].m128i_i64[0] = (__int64)v141;
                    v149 = 260;
                    v145[0].m128i_i64[0] = (__int64)",fillcolor=\"";
                    v147 = 1;
                    v146 = 3;
                    sub_9C6370(&v150, v145, v148, v24, v25, v26);
                    sub_9C6370(&v157, &v150, &src, v27, v28, v29);
                    sub_CA0F50(v142[0].m128i_i64, (void **)&v157);
                    sub_2241490((unsigned __int64 *)&v127, (char *)v142[0].m128i_i64[0], v142[0].m128i_u64[1]);
                    sub_2240A30((unsigned __int64 *)v142);
                    sub_2240A30((unsigned __int64 *)v141);
                    if ( *(_QWORD *)(v117 + 120) )
                    {
                      sub_2241520((unsigned __int64 *)&v127, ",color=\"blue\"");
                      sub_2241520((unsigned __int64 *)&v127, ",style=\"filled,bold,dashed\"");
                    }
                    else
                    {
                      sub_2241520((unsigned __int64 *)&v127, ",style=\"filled\"");
                    }
                    sub_2342640((__int64)v129);
                    v30 = sub_904010((__int64)&p_dest, "\tNode");
                    v31 = sub_CB5A80(v30, v117);
                    sub_904010(v31, " [shape=");
                    sub_904010((__int64)&p_dest, "record,");
                    if ( v128 )
                    {
                      v89 = sub_CB6200((__int64)&p_dest, v127, v128);
                      sub_904010(v89, ",");
                    }
                    sub_904010((__int64)&p_dest, "label=");
                    sub_904010((__int64)&p_dest, "\"{");
                    sub_2644B30(src.m128i_i64, v117, a1, v32, v33);
                    sub_C67200(v157.m128i_i64, (__int64)&src);
                    sub_CB6200((__int64)&p_dest, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
                    sub_2240A30((unsigned __int64 *)&v157);
                    sub_2240A30((unsigned __int64 *)&src);
                    sub_263F570(v150.m128i_i64, byte_3F871B3);
                    if ( v150.m128i_i64[1] )
                    {
                      v88 = sub_904010((__int64)&p_dest, "|");
                      sub_C67200(v157.m128i_i64, (__int64)&v150);
                      sub_CB6200(v88, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
                      sub_2240A30((unsigned __int64 *)&v157);
                    }
                    sub_263F570(src.m128i_i64, byte_3F871B3);
                    if ( src.m128i_i64[1] )
                    {
                      v87 = sub_904010((__int64)&p_dest, "|");
                      sub_C67200(v157.m128i_i64, (__int64)&src);
                      sub_CB6200(v87, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
                      sub_2240A30((unsigned __int64 *)&v157);
                    }
                    v34 = 0;
                    sub_2240A30((unsigned __int64 *)&src);
                    sub_2240A30((unsigned __int64 *)&v150);
                    v150.m128i_i64[0] = (__int64)&v151;
                    v160 = 0x100000000LL;
                    v161 = &v150;
                    v150.m128i_i64[1] = 0;
                    v157.m128i_i64[0] = (__int64)&unk_49DD210;
                    v151 = 0;
                    v157.m128i_i64[1] = 0;
                    v158 = 0u;
                    v159 = 0;
                    sub_CB5980((__int64)&v157, 0, 0, 0);
                    sub_904010((__int64)&p_dest, "}\"");
                    sub_904010((__int64)&p_dest, "];\n");
                    v36 = *(__int64 ***)(v117 + 48);
                    v37 = *(__int64 ***)(v117 + 56);
                    if ( v36 == v37 )
                      goto LABEL_55;
                    while ( 1 )
                    {
                      v41 = **v36;
                      if ( !*(_BYTE *)(v41 + 2) )
                        goto LABEL_46;
                      if ( dword_4FF3E28 == 1 )
                        break;
                      if ( dword_4FF3E28 == 2 )
                      {
                        sub_264D230((__int64)&src, **v36, v35);
                        if ( !v154 )
                          goto LABEL_119;
                        v75 = (v154 - 1) & (37 * qword_4FF3C68);
                        v76 = (_DWORD *)(src.m128i_i64[1] + 4LL * v75);
                        v77 = *v76;
                        if ( (_DWORD)qword_4FF3C68 != *v76 )
                        {
                          v82 = 1;
                          while ( v77 != -1 )
                          {
                            v99 = v82 + 1;
                            v75 = (v154 - 1) & (v82 + v75);
                            v76 = (_DWORD *)(src.m128i_i64[1] + 4LL * v75);
                            v77 = *v76;
                            if ( (_DWORD)qword_4FF3C68 == *v76 )
                              goto LABEL_108;
                            v82 = v99;
                          }
LABEL_119:
                          sub_2342640((__int64)&src);
                          goto LABEL_46;
                        }
LABEL_108:
                        v103 = v154;
                        v106 = src.m128i_i64[1];
                        v113 = v76;
                        sub_2342640((__int64)&src);
                        if ( v113 == (_DWORD *)(v106 + 4LL * v103) )
                          goto LABEL_46;
LABEL_53:
                        v41 = **v36;
                        if ( !v41 )
                        {
                          v36 += 2;
                          if ( v37 == v36 )
                          {
LABEL_55:
                            v157.m128i_i64[0] = (__int64)&unk_49DD210;
                            sub_CB5840((__int64)&v157);
                            sub_2240A30((unsigned __int64 *)&v150);
                            sub_2240A30((unsigned __int64 *)&v127);
                            goto LABEL_56;
                          }
                          goto LABEL_47;
                        }
                      }
                      v105 = v41;
                      sub_263F570(src.m128i_i64, byte_3F871B3);
                      v108 = src.m128i_i64[1];
                      sub_2240A30((unsigned __int64 *)&src);
                      sub_265E7F0(src.m128i_i64, (__int64 *)v36, a1);
                      v38 = sub_904010((__int64)&p_dest, "\tNode");
                      sub_CB5A80(v38, v117);
                      if ( v108 )
                      {
                        v39 = sub_904010((__int64)&p_dest, ":s");
                        sub_CB59F0(v39, v34);
                      }
                      v40 = sub_904010((__int64)&p_dest, " -> Node");
                      sub_CB5A80(v40, v105);
                      if ( src.m128i_i64[1] )
                      {
                        v85 = sub_904010((__int64)&p_dest, "[");
                        v86 = sub_CB6200(v85, (unsigned __int8 *)src.m128i_i64[0], src.m128i_u64[1]);
                        sub_904010(v86, "]");
                      }
                      sub_904010((__int64)&p_dest, ";\n");
                      sub_2240A30((unsigned __int64 *)&src);
LABEL_46:
                      v36 += 2;
                      if ( v37 == v36 )
                        goto LABEL_55;
LABEL_47:
                      if ( ++v34 == 64 )
                      {
                        while ( 1 )
                        {
                          v63 = **v36;
                          if ( !*(_BYTE *)(v63 + 2) )
                            goto LABEL_88;
                          if ( dword_4FF3E28 == 1 )
                            break;
                          if ( dword_4FF3E28 != 2 )
                            goto LABEL_83;
                          sub_264D230((__int64)&src, **v36, v35);
                          v78 = v154;
                          if ( !v154 )
                            goto LABEL_138;
                          v79 = (v154 - 1) & (37 * qword_4FF3C68);
                          v80 = (_DWORD *)(src.m128i_i64[1] + 4LL * v79);
                          v81 = *v80;
                          if ( *v80 != (_DWORD)qword_4FF3C68 )
                          {
                            v96 = 1;
                            while ( v81 != -1 )
                            {
                              v100 = v96 + 1;
                              v79 = (v154 - 1) & (v96 + v79);
                              v80 = (_DWORD *)(src.m128i_i64[1] + 4LL * v79);
                              v81 = *v80;
                              if ( (_DWORD)qword_4FF3C68 == *v80 )
                                goto LABEL_113;
                              v96 = v100;
                            }
LABEL_138:
                            sub_2342640((__int64)&src);
                            goto LABEL_88;
                          }
LABEL_113:
                          v107 = src.m128i_i64[1];
                          v114 = v80;
                          sub_2342640((__int64)&src);
                          if ( v114 != (_DWORD *)(v107 + 4 * v78) )
                            goto LABEL_94;
LABEL_88:
                          v36 += 2;
                          if ( v37 == v36 )
                            goto LABEL_55;
                        }
                        v112 = a1 + 96;
                        sub_264D230((__int64)&src, v63, v35);
                        if ( (unsigned int)v153 >= *(_DWORD *)(a1 + 112) )
                          v64 = sub_265E6F0(v112, (__int64)&src);
                        else
                          v64 = sub_265E6F0((__int64)&src, v112);
                        sub_2342640((__int64)&src);
                        if ( !v64 )
                          goto LABEL_88;
LABEL_94:
                        v63 = **v36;
                        if ( v63 )
                        {
LABEL_83:
                          sub_263F570(src.m128i_i64, byte_3F871B3);
                          v111 = src.m128i_i64[1];
                          sub_2240A30((unsigned __int64 *)&src);
                          sub_265E7F0(src.m128i_i64, (__int64 *)v36, a1);
                          v60 = sub_904010((__int64)&p_dest, "\tNode");
                          sub_CB5A80(v60, v117);
                          if ( v111 )
                          {
                            v61 = sub_904010((__int64)&p_dest, ":s");
                            sub_CB59F0(v61, 64);
                          }
                          v62 = sub_904010((__int64)&p_dest, " -> Node");
                          sub_CB5A80(v62, v63);
                          if ( src.m128i_i64[1] )
                          {
                            v97 = sub_904010((__int64)&p_dest, "[");
                            v98 = sub_CB6200(v97, (unsigned __int8 *)src.m128i_i64[0], src.m128i_u64[1]);
                            sub_904010(v98, "]");
                          }
                          sub_904010((__int64)&p_dest, ";\n");
                          sub_2240A30((unsigned __int64 *)&src);
                          goto LABEL_88;
                        }
                        goto LABEL_88;
                      }
                    }
                    v109 = a1 + 96;
                    sub_264D230((__int64)&src, v41, v35);
                    if ( (unsigned int)v153 >= *(_DWORD *)(a1 + 112) )
                      v42 = sub_265E6F0(v109, (__int64)&src);
                    else
                      v42 = sub_265E6F0((__int64)&src, v109);
                    v110 = v42;
                    sub_2342640((__int64)&src);
                    if ( !v110 )
                      goto LABEL_46;
                    goto LABEL_53;
                  }
                  v23 = "lightpink";
                  if ( byte_4FF31E8 )
                    goto LABEL_30;
                }
                v23 = "brown1";
                goto LABEL_30;
              }
              v101 = 1;
              while ( v95 != -1 )
              {
                v93 = (v158.m128i_i32[2] - 1) & (v101 + v93);
                v94 = (int *)(v157.m128i_i64[1] + 4LL * v93);
                v95 = *v94;
                if ( (_DWORD)qword_4FF3C68 == *v94 )
                  goto LABEL_133;
                ++v101;
              }
            }
            sub_2342640((__int64)&v157);
            goto LABEL_56;
          }
          v83 = a1 + 96;
          sub_264D230((__int64)&v157, v117, v10);
          if ( v158.m128i_i32[0] >= *(_DWORD *)(a1 + 112) )
            v84 = sub_265E6F0(v83, (__int64)&v157);
          else
            v84 = sub_265E6F0((__int64)&v157, v83);
          sub_2342640((__int64)&v157);
          if ( !v84 )
            goto LABEL_56;
          goto LABEL_24;
        }
LABEL_98:
        p_src = (__m128i *)&v125;
        v69 = sub_904010((__int64)&p_dest, "\tlabel=\"");
        goto LABEL_99;
      }
      v65 = &src;
      v66 = sub_904010((__int64)&p_dest, "digraph \"");
    }
    sub_C67200(v157.m128i_i64, (__int64)v65);
    v67 = sub_CB6200(v66, (unsigned __int8 *)v157.m128i_i64[0], v157.m128i_u64[1]);
    sub_904010(v67, "\" {\n");
    sub_2240A30((unsigned __int64 *)&v157);
    if ( !v126 )
      goto LABEL_19;
    goto LABEL_98;
  }
  v57 = sub_CB72A0();
  v58 = sub_904010((__int64)v57, "error opening file '");
  v59 = sub_CB6200(v58, (unsigned __int8 *)dest.m128i_i64[0], dest.m128i_u64[1]);
  sub_904010(v59, "' for writing!\n");
  v44 = byte_3F871B3;
  sub_263F570(v157.m128i_i64, byte_3F871B3);
LABEL_72:
  sub_CB5B00((int *)&p_dest, (__int64)v44);
LABEL_73:
  if ( (__m128i *)v157.m128i_i64[0] != &v158 )
    j_j___libc_free_0(v157.m128i_u64[0]);
  if ( (__m128i *)dest.m128i_i64[0] != &v124 )
    j_j___libc_free_0(dest.m128i_u64[0]);
  if ( (__m128i *)v121[0] != &v122 )
    j_j___libc_free_0(v121[0]);
  if ( (_QWORD *)v119[0] != v120 )
    j_j___libc_free_0(v119[0]);
}
