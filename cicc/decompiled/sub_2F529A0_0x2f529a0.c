// Function: sub_2F529A0
// Address: 0x2f529a0
//
void __fastcall sub_2F529A0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rbx
  _BYTE *v11; // r14
  _BYTE *v12; // rbx
  __int64 v13; // rax
  int *v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // r15d
  __int64 v21; // rax
  int *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rcx
  unsigned int *v28; // r15
  __int64 v29; // rax
  unsigned int *v30; // r12
  unsigned int *j; // r14
  int *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // r9
  int v39; // eax
  unsigned int v40; // ebx
  __int64 v41; // rax
  _QWORD *v42; // rdx
  int v43; // r8d
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rax
  int *v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // eax
  __int64 v55; // rdx
  unsigned int v56; // ecx
  __int64 v57; // rsi
  __int64 v58; // r13
  __int64 v59; // r15
  __int64 v60; // r12
  unsigned __int64 v61; // rcx
  __int64 v62; // rbx
  unsigned int v63; // eax
  __int64 v64; // rax
  unsigned int v65; // edx
  __int64 v66; // r10
  int v67; // r11d
  unsigned __int64 v68; // rcx
  unsigned int v69; // eax
  __int64 v70; // rbx
  unsigned int v71; // eax
  __int64 v72; // rsi
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // rdx
  __int64 v77; // rbx
  unsigned int v78; // ecx
  int v79; // eax
  int v80; // r10d
  unsigned __int64 v81; // r11
  _DWORD *v82; // rdx
  unsigned __int64 v83; // rcx
  unsigned __int64 v84; // rax
  __int64 v85; // rbx
  unsigned int v86; // edx
  __int64 v87; // r12
  unsigned __int64 v88; // rax
  _QWORD *v89; // rdx
  _QWORD *v90; // rdi
  const char *v91; // r12
  __int64 *v92; // rax
  int v93; // r12d
  int v94; // r10d
  unsigned __int64 v95; // r11
  _DWORD *v96; // rax
  unsigned __int64 v97; // rdx
  int v98; // r11d
  int v99; // r10d
  unsigned __int64 v100; // r12
  _DWORD *v101; // rax
  unsigned __int64 v102; // rdx
  __int64 v103; // [rsp+8h] [rbp-108h]
  unsigned int v104; // [rsp+10h] [rbp-100h]
  int v106; // [rsp+20h] [rbp-F0h]
  unsigned int v107; // [rsp+24h] [rbp-ECh]
  unsigned int v108; // [rsp+28h] [rbp-E8h]
  __int64 v109; // [rsp+28h] [rbp-E8h]
  __int64 v110; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v111; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v112; // [rsp+28h] [rbp-E8h]
  __int64 v113; // [rsp+30h] [rbp-E0h]
  unsigned int v114; // [rsp+30h] [rbp-E0h]
  __int64 v115; // [rsp+30h] [rbp-E0h]
  __int64 v116; // [rsp+30h] [rbp-E0h]
  int v117; // [rsp+30h] [rbp-E0h]
  int v118; // [rsp+30h] [rbp-E0h]
  int v119; // [rsp+30h] [rbp-E0h]
  int v120; // [rsp+30h] [rbp-E0h]
  __int64 v121; // [rsp+38h] [rbp-D8h]
  unsigned int v122; // [rsp+38h] [rbp-D8h]
  const void *v123; // [rsp+38h] [rbp-D8h]
  unsigned int v124; // [rsp+40h] [rbp-D0h]
  unsigned int v125; // [rsp+40h] [rbp-D0h]
  unsigned int v126; // [rsp+40h] [rbp-D0h]
  unsigned int *i; // [rsp+50h] [rbp-C0h]
  __int64 v130; // [rsp+50h] [rbp-C0h]
  unsigned __int8 v131; // [rsp+58h] [rbp-B8h]
  _QWORD *v132; // [rsp+58h] [rbp-B8h]
  int v133; // [rsp+58h] [rbp-B8h]
  __int64 v134; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v135; // [rsp+58h] [rbp-B8h]
  int v136; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v137[2]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE v138[32]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v139[2]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE v140[48]; // [rsp+A0h] [rbp-70h] BYREF
  int v141; // [rsp+D0h] [rbp-40h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 992);
  v107 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 8LL) - *(_DWORD *)(a2 + 64);
  v106 = *(_DWORD *)(*(_QWORD *)(v7 + 40) + 112LL);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 16LL * (v106 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v9 = 3LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL);
  v10 = *(_QWORD *)(a1 + 48) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v8 + 24LL);
  if ( *(_DWORD *)(a1 + 56) != *(_DWORD *)v10 )
  {
    sub_2F60630(a1 + 48, v8, v9, a4);
    v7 = *(_QWORD *)(a1 + 992);
  }
  v11 = *(_BYTE **)(v7 + 280);
  v131 = *(_BYTE *)(v10 + 8);
  v12 = &v11[40 * *(unsigned int *)(v7 + 288)];
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      v18 = 0;
      v20 = *(_DWORD *)(*(_QWORD *)v11 + 24LL);
      if ( !v11[32]
        || (v13 = *(unsigned int *)(*(_QWORD *)(v6 + 28800)
                                  + 4LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v6 + 824) + 8LL) + 8LL * v20)),
            (_DWORD)v13 == -1) )
      {
        v17 = 0;
        if ( !v11[33] )
          goto LABEL_47;
      }
      else
      {
        v14 = &dword_503BD90;
        v15 = *(_QWORD *)(v6 + 24176) + 144 * v13;
        v16 = *(_QWORD *)(v15 + 8);
        v17 = *(unsigned int *)(v15 + 4);
        if ( v16 )
        {
          v14 = (int *)(*(_QWORD *)(v16 + 512) + 24LL * v20);
          if ( *v14 != *(_DWORD *)(v16 + 4) )
          {
            v108 = *(_DWORD *)(v15 + 4);
            v113 = v15;
            sub_3501C20(v16);
            v17 = v108;
            v15 = v113;
            v14 = (int *)(*(_QWORD *)(v16 + 512) + 24LL * v20);
          }
        }
        *(_QWORD *)(v15 + 16) = v14;
        v18 = *((_QWORD *)v14 + 1);
        if ( !v11[33] )
        {
LABEL_10:
          if ( !(_DWORD)v17 )
            goto LABEL_47;
          v19 = *(_QWORD *)(v6 + 1000);
          goto LABEL_12;
        }
      }
      v21 = *(unsigned int *)(*(_QWORD *)(v6 + 28800)
                            + 4LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v6 + 824) + 8LL) + 4LL * (2 * v20 + 1)));
      if ( (_DWORD)v21 == -1 )
        goto LABEL_10;
      v22 = &dword_503BD90;
      v23 = *(_QWORD *)(v6 + 24176) + 144 * v21;
      v24 = *(_QWORD *)(v23 + 8);
      v25 = *(unsigned int *)(v23 + 4);
      if ( v24 )
      {
        v22 = (int *)(*(_QWORD *)(v24 + 512) + 24LL * v20);
        if ( *v22 != *(_DWORD *)(v24 + 4) )
        {
          v103 = v18;
          v104 = v17;
          v109 = v23;
          v114 = *(_DWORD *)(v23 + 4);
          v121 = *(_QWORD *)(v23 + 8);
          sub_3501C20(v24);
          v18 = v103;
          v17 = v104;
          v23 = v109;
          v25 = v114;
          v22 = (int *)(*(_QWORD *)(v121 + 512) + 24LL * v20);
        }
      }
      *(_QWORD *)(v23 + 16) = v22;
      v26 = *((_QWORD *)v22 + 2);
      if ( !((unsigned int)v25 | (unsigned int)v17) )
      {
LABEL_47:
        if ( (unsigned __int8)sub_2FB2C20(*(_QWORD *)(v6 + 992), v11, v131, v18) )
          sub_2FBE000(*(_QWORD *)(v6 + 1000), v11);
        goto LABEL_13;
      }
      v19 = *(_QWORD *)(v6 + 1000);
      if ( !(_DWORD)v17 )
      {
        sub_2FBDBC0(v19, v11, (unsigned int)v25, v26);
        goto LABEL_13;
      }
      if ( (_DWORD)v25 )
      {
        v11 += 40;
        sub_2FBE6C0(v19, v20, v17, v18, v25, v26);
        if ( v12 == v11 )
        {
LABEL_24:
          v7 = *(_QWORD *)(v6 + 992);
          break;
        }
      }
      else
      {
LABEL_12:
        sub_2FBE160(v19, v11, v17, v18);
LABEL_13:
        v11 += 40;
        if ( v12 == v11 )
          goto LABEL_24;
      }
    }
  }
  v27 = *(unsigned int *)(v7 + 632);
  v139[0] = (unsigned __int64)v140;
  v139[1] = 0x600000000LL;
  if ( (_DWORD)v27 )
    sub_2F4CBD0((__int64)v139, v7 + 624, v9, v27, a5, a6);
  v28 = a3;
  v141 = *(_DWORD *)(v7 + 688);
  for ( i = &a3[a4]; i != v28; ++v28 )
  {
    v29 = *(_QWORD *)(v6 + 24176) + 144LL * *v28;
    v30 = *(unsigned int **)(v29 + 96);
    for ( j = &v30[*(unsigned int *)(v29 + 104)]; j != v30; ++v30 )
    {
      v40 = *v30;
      v41 = 1LL << *v30;
      v42 = (_QWORD *)(v139[0] + 8LL * (*v30 >> 6));
      if ( (*v42 & v41) != 0 )
      {
        v43 = 2 * v40;
        v38 = 0;
        v36 = 0;
        *v42 &= ~v41;
        v37 = 0;
        v44 = *(_QWORD *)(v6 + 28800);
        v45 = *(_QWORD *)(*(_QWORD *)(v6 + 824) + 8LL);
        v46 = *(unsigned int *)(v44 + 4LL * *(unsigned int *)(v45 + 8LL * v40));
        if ( (_DWORD)v46 != -1 )
        {
          v47 = &dword_503BD90;
          v48 = *(_QWORD *)(v6 + 24176) + 144 * v46;
          v49 = *(_QWORD *)(v48 + 8);
          v37 = *(unsigned int *)(v48 + 4);
          if ( v49 )
          {
            v47 = (int *)(*(_QWORD *)(v49 + 512) + 24LL * v40);
            if ( *v47 != *(_DWORD *)(v49 + 4) )
            {
              v116 = v48;
              v125 = *(_DWORD *)(v48 + 4);
              sub_3501C20(v49);
              v38 = 0;
              v48 = v116;
              v43 = 2 * v40;
              v37 = v125;
              v47 = (int *)(*(_QWORD *)(v49 + 512) + 24LL * v40);
            }
          }
          *(_QWORD *)(v48 + 16) = v47;
          v36 = *((_QWORD *)v47 + 1);
          v44 = *(_QWORD *)(v6 + 28800);
          v45 = *(_QWORD *)(*(_QWORD *)(v6 + 824) + 8LL);
        }
        v50 = *(unsigned int *)(v44 + 4LL * *(unsigned int *)(v45 + 4LL * (unsigned int)(v43 + 1)));
        if ( (_DWORD)v50 == -1 )
        {
          v39 = v37;
          v35 = 0;
        }
        else
        {
          v32 = &dword_503BD90;
          v33 = *(_QWORD *)(v6 + 24176) + 144 * v50;
          v34 = *(_QWORD *)(v33 + 8);
          v35 = *(unsigned int *)(v33 + 4);
          if ( v34 )
          {
            v32 = (int *)(*(_QWORD *)(v34 + 512) + 24LL * v40);
            if ( *v32 != *(_DWORD *)(v34 + 4) )
            {
              v110 = v36;
              v115 = v33;
              v122 = *(_DWORD *)(v33 + 4);
              v124 = v37;
              sub_3501C20(v34);
              v36 = v110;
              v33 = v115;
              v35 = v122;
              v37 = v124;
              v32 = (int *)(*(_QWORD *)(v34 + 512) + 24LL * v40);
            }
          }
          *(_QWORD *)(v33 + 16) = v32;
          v38 = *((_QWORD *)v32 + 2);
          v39 = v35 | v37;
        }
        if ( v39 )
          sub_2FBE6C0(*(_QWORD *)(v6 + 1000), v40, v37, v36, v35, v38);
      }
    }
  }
  v51 = *(_QWORD *)(v6 + 1000);
  v137[0] = (unsigned __int64)v138;
  v137[1] = 0x800000000LL;
  sub_2FBB760(v51, v137);
  sub_2E01430(
    *(__int64 **)(v6 + 840),
    v106,
    (unsigned int *)(**(_QWORD **)(a2 + 16) + 4LL * *(unsigned int *)(a2 + 64)),
    *(unsigned int *)(*(_QWORD *)(a2 + 16) + 8LL) - (unsigned __int64)*(unsigned int *)(a2 + 64));
  v54 = *(_DWORD *)(a2 + 64);
  v126 = *(_DWORD *)(*(_QWORD *)(v6 + 992) + 288LL)
       + *(_DWORD *)(*(_QWORD *)(v6 + 992) + 696LL)
       - *(_DWORD *)(*(_QWORD *)(v6 + 992) + 616LL);
  v55 = *(_QWORD *)(a2 + 16);
  v56 = *(_DWORD *)(v55 + 8) - v54;
  if ( v56 )
  {
    v130 = v56;
    v123 = (const void *)(v6 + 936);
    v57 = v6;
    v58 = 0;
    v59 = v57;
    while ( 1 )
    {
      v66 = *(_QWORD *)(v59 + 32);
      v67 = *(_DWORD *)(*(_QWORD *)v55 + 4LL * (unsigned int)(v54 + v58));
      v68 = *(unsigned int *)(v66 + 160);
      v69 = v67 & 0x7FFFFFFF;
      v70 = 8LL * (v67 & 0x7FFFFFFF);
      if ( (v67 & 0x7FFFFFFFu) >= (unsigned int)v68 )
        break;
      v60 = *(_QWORD *)(*(_QWORD *)(v66 + 152) + 8LL * v69);
      if ( !v60 )
        break;
LABEL_53:
      v61 = *(unsigned int *)(v59 + 928);
      v62 = *(_DWORD *)(v60 + 112) & 0x7FFFFFFF;
      v63 = v62 + 1;
      if ( (int)v62 + 1 > (unsigned int)v61 )
      {
        v74 = v63;
        if ( v63 != v61 )
        {
          if ( v63 >= v61 )
          {
            v79 = *(_DWORD *)(v59 + 936);
            v80 = *(_DWORD *)(v59 + 940);
            v81 = v74 - v61;
            if ( v74 > *(unsigned int *)(v59 + 932) )
            {
              v111 = v74 - v61;
              v117 = *(_DWORD *)(v59 + 940);
              v133 = *(_DWORD *)(v59 + 936);
              sub_C8D5F0(v59 + 920, v123, v74, 8u, v52, v53);
              v61 = *(unsigned int *)(v59 + 928);
              v81 = v111;
              v80 = v117;
              v79 = v133;
            }
            v82 = (_DWORD *)(*(_QWORD *)(v59 + 920) + 8 * v61);
            v83 = v81;
            do
            {
              if ( v82 )
              {
                *v82 = v79;
                v82[1] = v80;
              }
              v82 += 2;
              --v83;
            }
            while ( v83 );
            *(_DWORD *)(v59 + 928) += v81;
          }
          else
          {
            *(_DWORD *)(v59 + 928) = v63;
          }
        }
      }
      v64 = *(_QWORD *)(v59 + 920);
      if ( *(_DWORD *)(v64 + 8 * v62) )
        goto LABEL_57;
      v65 = *(_DWORD *)(v137[0] + 4 * v58);
      if ( v65 )
      {
        if ( v65 < v107 && (unsigned int)sub_2FB1B30(*(_QWORD *)(v59 + 992), v60) >= v126 )
        {
          v84 = *(unsigned int *)(v59 + 928);
          v85 = *(_DWORD *)(v60 + 112) & 0x7FFFFFFF;
          v86 = v85 + 1;
          if ( (int)v85 + 1 > (unsigned int)v84 )
          {
            v52 = v86;
            if ( v86 != v84 )
            {
              if ( v86 >= v84 )
              {
                v98 = *(_DWORD *)(v59 + 936);
                v99 = *(_DWORD *)(v59 + 940);
                v100 = v86 - v84;
                if ( v86 > (unsigned __int64)*(unsigned int *)(v59 + 932) )
                {
                  v120 = *(_DWORD *)(v59 + 936);
                  v136 = *(_DWORD *)(v59 + 940);
                  sub_C8D5F0(v59 + 920, v123, v86, 8u, v86, v53);
                  v84 = *(unsigned int *)(v59 + 928);
                  v98 = v120;
                  v99 = v136;
                }
                v101 = (_DWORD *)(*(_QWORD *)(v59 + 920) + 8 * v84);
                v102 = v100;
                do
                {
                  if ( v101 )
                  {
                    *v101 = v98;
                    v101[1] = v99;
                  }
                  v101 += 2;
                  --v102;
                }
                while ( v102 );
                *(_DWORD *)(v59 + 928) += v100;
              }
              else
              {
                *(_DWORD *)(v59 + 928) = v86;
              }
            }
          }
          *(_DWORD *)(*(_QWORD *)(v59 + 920) + 8 * v85) = 3;
        }
LABEL_57:
        if ( ++v58 == v130 )
          goto LABEL_71;
        goto LABEL_58;
      }
      v76 = *(unsigned int *)(v59 + 928);
      v77 = *(_DWORD *)(v60 + 112) & 0x7FFFFFFF;
      v78 = v77 + 1;
      if ( (int)v77 + 1 > (unsigned int)v76 )
      {
        v52 = v78;
        if ( v78 != v76 )
        {
          if ( v78 >= v76 )
          {
            v93 = *(_DWORD *)(v59 + 936);
            v94 = *(_DWORD *)(v59 + 940);
            v95 = v78 - v76;
            if ( v78 > (unsigned __int64)*(unsigned int *)(v59 + 932) )
            {
              v119 = *(_DWORD *)(v59 + 940);
              v135 = v78 - v76;
              sub_C8D5F0(v59 + 920, v123, v78, 8u, v78, v53);
              v64 = *(_QWORD *)(v59 + 920);
              v76 = *(unsigned int *)(v59 + 928);
              v94 = v119;
              v95 = v135;
            }
            v96 = (_DWORD *)(v64 + 8 * v76);
            v97 = v95;
            do
            {
              if ( v96 )
              {
                *v96 = v93;
                v96[1] = v94;
              }
              v96 += 2;
              --v97;
            }
            while ( v97 );
            *(_DWORD *)(v59 + 928) += v95;
            v64 = *(_QWORD *)(v59 + 920);
          }
          else
          {
            *(_DWORD *)(v59 + 928) = v78;
          }
        }
      }
      *(_DWORD *)(v64 + 8 * v77) = 4;
      if ( ++v58 == v130 )
      {
LABEL_71:
        v6 = v59;
        goto LABEL_72;
      }
LABEL_58:
      v55 = *(_QWORD *)(a2 + 16);
      v54 = *(_DWORD *)(a2 + 64);
    }
    v71 = v69 + 1;
    if ( (unsigned int)v68 < v71 )
    {
      v75 = v71;
      if ( v71 != v68 )
      {
        if ( v71 >= v68 )
        {
          v87 = *(_QWORD *)(v66 + 168);
          v88 = v71 - v68;
          if ( v75 > *(unsigned int *)(v66 + 164) )
          {
            v112 = v88;
            v118 = v67;
            v134 = *(_QWORD *)(v59 + 32);
            sub_C8D5F0(v66 + 152, (const void *)(v66 + 168), v75, 8u, v52, v53);
            v66 = v134;
            v88 = v112;
            v67 = v118;
            v68 = *(unsigned int *)(v134 + 160);
          }
          v72 = *(_QWORD *)(v66 + 152);
          v89 = (_QWORD *)(v72 + 8 * v68);
          v90 = &v89[v88];
          if ( v89 != v90 )
          {
            do
              *v89++ = v87;
            while ( v90 != v89 );
            LODWORD(v68) = *(_DWORD *)(v66 + 160);
            v72 = *(_QWORD *)(v66 + 152);
          }
          *(_DWORD *)(v66 + 160) = v88 + v68;
          goto LABEL_62;
        }
        *(_DWORD *)(v66 + 160) = v71;
      }
    }
    v72 = *(_QWORD *)(v66 + 152);
LABEL_62:
    v132 = (_QWORD *)v66;
    v73 = sub_2E10F30(v67);
    *(_QWORD *)(v72 + v70) = v73;
    v60 = v73;
    sub_2E11E80(v132, v73);
    goto LABEL_53;
  }
LABEL_72:
  if ( unk_503FCFD )
  {
    v91 = *(const char **)(v6 + 768);
    v92 = (__int64 *)sub_CB72A0();
    sub_2F06090(
      v91,
      *(_QWORD *)(v6 + 32),
      *(_QWORD *)(v6 + 784),
      (__int64)"After splitting live range around region",
      v92,
      1);
  }
  if ( (_BYTE *)v137[0] != v138 )
    _libc_free(v137[0]);
  if ( (_BYTE *)v139[0] != v140 )
    _libc_free(v139[0]);
}
