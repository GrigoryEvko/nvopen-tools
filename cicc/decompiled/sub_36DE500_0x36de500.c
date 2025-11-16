// Function: sub_36DE500
// Address: 0x36de500
//
__int64 __fastcall sub_36DE500(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r12
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // r15d
  unsigned __int64 *v14; // rcx
  __int64 v15; // r14
  unsigned __int64 **v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 *v19; // r14
  unsigned __int8 *v20; // rax
  __int64 v21; // rdi
  __int32 v22; // edx
  unsigned __int8 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int32 v26; // edx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  __m128i v30; // xmm2
  int v31; // eax
  unsigned __int16 v32; // r13
  unsigned __int64 v33; // rdx
  int v34; // eax
  __int16 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdi
  unsigned __int16 v38; // dx
  __int64 v39; // r8
  unsigned __int8 *v40; // rax
  __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  unsigned __int64 *v44; // rdx
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rax
  __int16 v48; // ax
  __int128 v49; // rax
  __int64 v50; // r9
  __int64 v51; // rax
  unsigned __int64 v52; // r8
  unsigned __int64 *v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rax
  _QWORD *v58; // rdi
  __int64 v59; // r13
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int128 v64; // rax
  __int64 v65; // r9
  __int64 v66; // rax
  __int64 v67; // rdi
  unsigned int i; // eax
  __int64 v69; // rdx
  __int64 v70; // r13
  __int64 v71; // r10
  __int64 v72; // r12
  __int64 v73; // r11
  __int64 v74; // rax
  __int64 v75; // rdx
  unsigned __int64 *v76; // rbx
  unsigned __int64 v77; // rcx
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // r8
  __int64 *v81; // r11
  char v82; // al
  char v83; // si
  char v84; // dl
  char v85; // cl
  char v86; // al
  char v87; // al
  char v88; // si
  char v89; // dl
  char v90; // cl
  char v91; // al
  char v92; // si
  char v93; // dl
  char v94; // cl
  char v95; // al
  char v96; // si
  char v97; // dl
  char v98; // cl
  char v99; // al
  __int64 v100; // rdx
  __int64 v101; // [rsp+10h] [rbp-1F0h]
  __int64 v102; // [rsp+18h] [rbp-1E8h]
  __int64 v103; // [rsp+20h] [rbp-1E0h]
  _QWORD *v104; // [rsp+28h] [rbp-1D8h]
  __m128i v105; // [rsp+30h] [rbp-1D0h] BYREF
  unsigned __int64 **v106; // [rsp+40h] [rbp-1C0h]
  __int64 v107; // [rsp+48h] [rbp-1B8h]
  __int64 *v108; // [rsp+50h] [rbp-1B0h]
  __int64 v109; // [rsp+58h] [rbp-1A8h]
  __int64 v110; // [rsp+60h] [rbp-1A0h]
  unsigned __int64 *v111; // [rsp+68h] [rbp-198h]
  __int64 v112; // [rsp+70h] [rbp-190h]
  __int64 v113; // [rsp+78h] [rbp-188h]
  __int64 v114; // [rsp+80h] [rbp-180h]
  __int64 v115; // [rsp+88h] [rbp-178h]
  __int64 v116; // [rsp+90h] [rbp-170h]
  __int64 v117; // [rsp+98h] [rbp-168h]
  unsigned __int64 v118; // [rsp+A0h] [rbp-160h]
  __int64 v119; // [rsp+A8h] [rbp-158h]
  unsigned __int8 *v120; // [rsp+B0h] [rbp-150h]
  __int64 v121; // [rsp+B8h] [rbp-148h]
  __int64 v122; // [rsp+C0h] [rbp-140h]
  __int64 v123; // [rsp+C8h] [rbp-138h]
  unsigned __int64 v124; // [rsp+D8h] [rbp-128h]
  __int64 v125; // [rsp+E0h] [rbp-120h] BYREF
  int v126; // [rsp+E8h] [rbp-118h]
  __int64 v127; // [rsp+F0h] [rbp-110h] BYREF
  int v128; // [rsp+F8h] [rbp-108h]
  __m128i v129; // [rsp+100h] [rbp-100h] BYREF
  __m128i v130; // [rsp+110h] [rbp-F0h] BYREF
  __m128i v131; // [rsp+120h] [rbp-E0h] BYREF
  __m128i v132; // [rsp+130h] [rbp-D0h] BYREF
  unsigned __int64 *v133; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v134; // [rsp+148h] [rbp-B8h]
  _BYTE v135[176]; // [rsp+150h] [rbp-B0h] BYREF

  v3 = a2;
  v5 = *(_QWORD *)(a2 + 80);
  v125 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v125, v5, 1);
  v6 = *(_QWORD *)(v3 + 40);
  v126 = *(_DWORD *)(v3 + 72);
  v110 = *(_QWORD *)v6;
  LODWORD(v108) = *(_DWORD *)(v6 + 8);
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 96LL);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = *(_QWORD *)(*(_QWORD *)(v6 + 80) + 96LL);
  if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    v109 = *(_QWORD *)(v9 + 24);
  else
    v109 = **(_QWORD **)(v9 + 24);
  v10 = v6 + 40LL * (unsigned int)(*(_DWORD *)(v3 + 64) - 1);
  v11 = *(_QWORD *)v10;
  LODWORD(v107) = *(_DWORD *)(v10 + 8);
  v12 = (unsigned int)(*(_DWORD *)(v3 + 24) - 571);
  if ( (unsigned int)v12 > 4 )
    goto LABEL_154;
  v13 = dword_4500FF0[v12];
  v111 = (unsigned __int64 *)v135;
  v133 = (unsigned __int64 *)v135;
  v134 = 0x800000000LL;
  if ( v13 )
  {
    a3 = _mm_loadu_si128((const __m128i *)(v6 + 120));
    v14 = v111;
    v15 = 160;
    v16 = &v133;
    v17 = 40LL * (v13 - 1) + 160;
    v18 = 0;
    while ( 1 )
    {
      *(__m128i *)&v14[2 * v18] = a3;
      v18 = (unsigned int)(v134 + 1);
      LODWORD(v134) = v134 + 1;
      if ( v17 == v15 )
        break;
      a3 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v3 + 40) + v15));
      if ( v18 + 1 > (unsigned __int64)HIDWORD(v134) )
      {
        v103 = v17;
        v104 = v8;
        v106 = v16;
        v105 = a3;
        sub_C8D5F0((__int64)v16, v111, v18 + 1, 0x10u, v17, v18 + 1);
        v18 = (unsigned int)v134;
        v17 = v103;
        v8 = v104;
        a3 = _mm_load_si128(&v105);
        v16 = v106;
      }
      v14 = v133;
      v15 += 40;
    }
  }
  v19 = &v125;
  v20 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v8, (__int64)&v125, 7, 0, 1u, a3, 0);
  v21 = *(_QWORD *)(a1 + 64);
  v129.m128i_i32[2] = v22;
  v129.m128i_i64[0] = (__int64)v20;
  v23 = sub_3400BD0(v21, (unsigned int)v109, (__int64)&v125, 7, 0, 1u, a3, 0);
  v132.m128i_i64[0] = v11;
  v130.m128i_i64[0] = (__int64)v23;
  v130.m128i_i32[2] = v26;
  v131.m128i_i64[0] = v110;
  v131.m128i_i32[2] = (int)v108;
  v132.m128i_i32[2] = v107;
  v27 = (unsigned int)v134;
  v28 = (unsigned int)v134 + 4LL;
  if ( v28 > HIDWORD(v134) )
  {
    sub_C8D5F0((__int64)&v133, v111, v28, 0x10u, v24, v25);
    v27 = (unsigned int)v134;
  }
  v29 = &v133[2 * v27];
  *(__m128i *)v29 = _mm_load_si128(&v129);
  v30 = _mm_load_si128(&v130);
  LODWORD(v134) = v134 + 4;
  *((__m128i *)v29 + 1) = v30;
  *((__m128i *)v29 + 2) = _mm_load_si128(&v131);
  *((__m128i *)v29 + 3) = _mm_load_si128(&v132);
  v31 = *(_DWORD *)(v3 + 24);
  if ( v31 == 574 )
  {
    *(_QWORD *)&v49 = sub_3400BD0(*(_QWORD *)(a1 + 64), 0, (__int64)&v125, 7, 0, 1u, a3, 0);
    v51 = sub_33F77A0(*(_QWORD **)(a1 + 64), 1141, (__int64)&v125, 7u, 0, v50, *(_OWORD *)v133, v49);
LABEL_33:
    LODWORD(v46) = 4650;
    v52 = v51;
    v53 = v133;
    *v133 = v52;
    *((_DWORD *)v53 + 2) = 0;
    goto LABEL_34;
  }
  if ( v31 == 575 )
  {
    *(_QWORD *)&v64 = sub_3400BD0(*(_QWORD *)(a1 + 64), 0, (__int64)&v125, 7, 0, 1u, a3, 0);
    v51 = sub_33F77A0(*(_QWORD **)(a1 + 64), 1203, (__int64)&v125, 7u, 0, v65, *(_OWORD *)v133, v64);
    goto LABEL_33;
  }
  if ( ((v13 - 2) & 0xFFFFFFFD) != 0 )
  {
    if ( v13 == 1 )
    {
      v32 = *(_WORD *)(v3 + 96);
      if ( v32 == 11
        || v32 == 127
        || (v33 = *v133, v34 = *(_DWORD *)(*v133 + 24), (unsigned int)(v34 - 35) > 1) && (unsigned int)(v34 - 11) > 1 )
      {
        v129.m128i_i64[0] = 0x100001226LL;
        v127 = 0x10000122CLL;
        v45 = sub_36D6650(v32, 4656, 4648, 4650, 0x10000122CLL, 4644, 0x100001226LL);
      }
      else
      {
        v35 = *(__int16 **)(v33 + 48);
        v36 = *(_QWORD *)(v33 + 96);
        v37 = *(_QWORD *)(a1 + 64);
        v38 = *v35;
        v39 = *((_QWORD *)v35 + 1);
        if ( (unsigned __int16)(v32 - 12) <= 1u )
        {
          v122 = sub_33FE020(v37, v36, (__int64)&v125, v38, v39, 1, a3);
          v42 = v122;
          v43 = (unsigned int)v100;
          v123 = v100;
        }
        else
        {
          v40 = sub_33FF780(v37, v36, (__int64)&v125, v38, v39, 1u, a3, 0);
          v121 = v41;
          v42 = (unsigned __int64)v40;
          v120 = v40;
          v43 = (unsigned int)v41;
        }
        v44 = v133;
        v118 = v42;
        v119 = v43;
        *v133 = v42;
        *((_DWORD *)v44 + 2) = v119;
        v129.m128i_i64[0] = 0x100001225LL;
        v127 = 0x10000122BLL;
        v45 = sub_36D6650(v32, 4655, 4647, 4649, 0x10000122BLL, 4643, 0x100001225LL);
      }
      v46 = v45;
      v47 = HIDWORD(v45);
      v124 = v46;
      if ( (_DWORD)v46 == 4656 )
      {
        if ( (_BYTE)v47 )
        {
          LODWORD(v46) = 4653;
          v48 = *(_WORD *)(*(_QWORD *)(*v133 + 48) + 16LL * *((unsigned int *)v133 + 2));
          if ( v48 != 7 )
            LODWORD(v46) = 2 * (v48 != 8) + 4654;
        }
      }
      goto LABEL_34;
    }
LABEL_154:
    BUG();
  }
  LOWORD(v110) = *(_WORD *)(v3 + 96);
  v127 = v125;
  if ( v125 )
    sub_B96E90((__int64)&v127, v125, 1);
  v130.m128i_i64[0] = 4;
  v108 = &v130.m128i_i64[1];
  v128 = v126;
  v66 = *(_QWORD *)(a1 + 64);
  v129.m128i_i64[0] = (__int64)&v130.m128i_i64[1];
  v67 = v66;
  if ( v13 )
  {
    for ( i = 0; i < v13; ++i )
    {
      v69 = i;
      v130.m128i_i8[v69 + 8] = 0;
    }
  }
  v106 = (unsigned __int64 **)v3;
  v70 = 0;
  v105.m128i_i64[0] = (__int64)&v125;
  v71 = v101;
  v72 = (__int64)v108;
  v129.m128i_i64[1] = v13;
  v73 = v102;
  v107 = a1;
  do
  {
    v77 = v133[2 * v70];
    *(_BYTE *)(v72 + v70) = (unsigned int)(*(_DWORD *)(v77 + 24) - 35) <= 1
                         || (unsigned int)(*(_DWORD *)(v77 + 24) - 11) <= 1;
    v72 = v129.m128i_i64[0];
    if ( *(_BYTE *)(v129.m128i_i64[0] + v70) )
    {
      v78 = *(_QWORD *)(v77 + 48);
      v79 = *(_QWORD *)(v77 + 96);
      v80 = *(_QWORD *)(v78 + 8);
      if ( (unsigned __int16)(v110 - 12) > 1u )
      {
        LOWORD(v71) = *(_WORD *)v78;
        v109 = v73;
        v74 = (__int64)sub_33FF780(v67, v79, (__int64)&v127, (unsigned int)v71, v80, 1u, a3, 0);
        v73 = v109;
        v114 = v74;
        v115 = v75;
        v75 = (unsigned int)v75;
      }
      else
      {
        LOWORD(v73) = *(_WORD *)v78;
        v109 = v71;
        v74 = sub_33FE020(v67, v79, (__int64)&v127, (unsigned int)v73, v80, 1, a3);
        v71 = v109;
        v117 = v75;
        v75 = (unsigned int)v75;
        v116 = v74;
      }
      v76 = &v133[2 * v70];
      v112 = v74;
      v113 = v75;
      *v76 = v74;
      *((_DWORD *)v76 + 2) = v113;
    }
    ++v70;
  }
  while ( v13 > (unsigned int)v70 );
  v81 = (__int64 *)v72;
  a1 = v107;
  v3 = (__int64)v106;
  v19 = (__int64 *)v105.m128i_i64[0];
  if ( (unsigned __int16)v110 <= 0x2Fu )
  {
    if ( (unsigned __int16)v110 > 1u )
    {
      switch ( (__int16)v110 )
      {
        case 2:
          LODWORD(v46) = 4744;
          if ( v13 == 2 )
            LODWORD(v46) = 4680;
          goto LABEL_59;
        case 5:
          v96 = *(_BYTE *)v81;
          v97 = *((_BYTE *)v81 + 1);
          if ( v13 == 2 )
          {
            if ( v96 )
              LODWORD(v46) = (v97 == 0) + 4677;
            else
              LODWORD(v46) = (v97 == 0) + 4679;
          }
          else
          {
            v98 = *((_BYTE *)v81 + 2);
            v99 = *((_BYTE *)v81 + 3);
            if ( v96 )
            {
              if ( v97 )
              {
                if ( v98 )
                  LODWORD(v46) = (v99 == 0) + 4729;
                else
                  LODWORD(v46) = (v99 == 0) + 4731;
              }
              else if ( v98 )
              {
                LODWORD(v46) = (v99 == 0) + 4733;
              }
              else
              {
                LODWORD(v46) = (v99 == 0) + 4735;
              }
            }
            else if ( v97 )
            {
              if ( v98 )
                LODWORD(v46) = (v99 == 0) + 4737;
              else
                LODWORD(v46) = (v99 == 0) + 4739;
            }
            else if ( v98 )
            {
              LODWORD(v46) = (v99 == 0) + 4741;
            }
            else
            {
              LODWORD(v46) = (v99 == 0) + 4743;
            }
          }
          goto LABEL_59;
        case 6:
          v92 = *(_BYTE *)v81;
          v93 = *((_BYTE *)v81 + 1);
          if ( v13 == 2 )
          {
            if ( v92 )
              LODWORD(v46) = (v93 == 0) + 4665;
            else
              LODWORD(v46) = (v93 == 0) + 4667;
          }
          else
          {
            v94 = *((_BYTE *)v81 + 2);
            v95 = *((_BYTE *)v81 + 3);
            if ( v92 )
            {
              if ( v93 )
              {
                if ( v94 )
                  LODWORD(v46) = (v95 == 0) + 4697;
                else
                  LODWORD(v46) = (v95 == 0) + 4699;
              }
              else if ( v94 )
              {
                LODWORD(v46) = (v95 == 0) + 4701;
              }
              else
              {
                LODWORD(v46) = (v95 == 0) + 4703;
              }
            }
            else if ( v93 )
            {
              if ( v94 )
                LODWORD(v46) = (v95 == 0) + 4705;
              else
                LODWORD(v46) = (v95 == 0) + 4707;
            }
            else if ( v94 )
            {
              LODWORD(v46) = (v95 == 0) + 4709;
            }
            else
            {
              LODWORD(v46) = (v95 == 0) + 4711;
            }
          }
          goto LABEL_59;
        case 7:
          v88 = *(_BYTE *)v81;
          v89 = *((_BYTE *)v81 + 1);
          if ( v13 == 2 )
          {
            if ( v88 )
              LODWORD(v46) = (v89 == 0) + 4669;
            else
              LODWORD(v46) = (v89 == 0) + 4671;
          }
          else
          {
            v90 = *((_BYTE *)v81 + 2);
            v91 = *((_BYTE *)v81 + 3);
            if ( v88 )
            {
              if ( v89 )
              {
                if ( v90 )
                  LODWORD(v46) = (v91 == 0) + 4713;
                else
                  LODWORD(v46) = (v91 == 0) + 4715;
              }
              else if ( v90 )
              {
                LODWORD(v46) = (v91 == 0) + 4717;
              }
              else
              {
                LODWORD(v46) = (v91 == 0) + 4719;
              }
            }
            else if ( v89 )
            {
              if ( v90 )
                LODWORD(v46) = (v91 == 0) + 4721;
              else
                LODWORD(v46) = (v91 == 0) + 4723;
            }
            else if ( v90 )
            {
              LODWORD(v46) = (v91 == 0) + 4725;
            }
            else
            {
              LODWORD(v46) = (v91 == 0) + 4727;
            }
          }
          goto LABEL_59;
        case 8:
          v87 = *((_BYTE *)v81 + 1);
          if ( *(_BYTE *)v81 )
            LODWORD(v46) = (v87 == 0) + 4673;
          else
            LODWORD(v46) = (v87 == 0) + 4675;
          goto LABEL_59;
        case 10:
        case 11:
          LODWORD(v46) = 4668;
          if ( v13 != 2 )
            LODWORD(v46) = 4712;
          goto LABEL_59;
        case 12:
          v83 = *(_BYTE *)v81;
          v84 = *((_BYTE *)v81 + 1);
          if ( v13 == 2 )
          {
            if ( v83 )
              LODWORD(v46) = (v84 == 0) + 4657;
            else
              LODWORD(v46) = (v84 == 0) + 4659;
          }
          else
          {
            v85 = *((_BYTE *)v81 + 2);
            v86 = *((_BYTE *)v81 + 3);
            if ( v83 )
            {
              if ( v84 )
              {
                if ( v85 )
                  LODWORD(v46) = (v86 == 0) + 4681;
                else
                  LODWORD(v46) = (v86 == 0) + 4683;
              }
              else if ( v85 )
              {
                LODWORD(v46) = (v86 == 0) + 4685;
              }
              else
              {
                LODWORD(v46) = (v86 == 0) + 4687;
              }
            }
            else if ( v84 )
            {
              if ( v85 )
                LODWORD(v46) = (v86 == 0) + 4689;
              else
                LODWORD(v46) = (v86 == 0) + 4691;
            }
            else if ( v85 )
            {
              LODWORD(v46) = (v86 == 0) + 4693;
            }
            else
            {
              LODWORD(v46) = (v86 == 0) + 4695;
            }
          }
          goto LABEL_59;
        case 13:
          v82 = *((_BYTE *)v81 + 1);
          if ( *(_BYTE *)v81 )
            LODWORD(v46) = (v82 == 0) + 4661;
          else
            LODWORD(v46) = (v82 == 0) + 4663;
          goto LABEL_59;
        case 37:
        case 47:
          goto LABEL_57;
        default:
          goto LABEL_154;
      }
    }
    goto LABEL_154;
  }
  if ( (_WORD)v110 != 127 && (_WORD)v110 != 138 )
    goto LABEL_154;
LABEL_57:
  LODWORD(v46) = 4672;
  if ( v13 != 2 )
    LODWORD(v46) = 4728;
LABEL_59:
  if ( v81 != v108 )
    _libc_free((unsigned __int64)v81);
  if ( v127 )
    sub_B91220((__int64)&v127, v127);
LABEL_34:
  v54 = sub_33E5110(*(__int64 **)(a1 + 64), 1, 0, 262, 0);
  v57 = sub_33E66D0(*(_QWORD **)(a1 + 64), v46, (__int64)v19, v54, v55, v56, v133, (unsigned int)v134);
  v58 = *(_QWORD **)(a1 + 64);
  v59 = v57;
  v129.m128i_i64[0] = *(_QWORD *)(v3 + 112);
  sub_33E4DA0(v58, v57, v129.m128i_i64, 1);
  sub_34158F0(*(_QWORD *)(a1 + 64), v3, v59, v60, v61, v62);
  sub_3421DB0(v59);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), v3);
  if ( v133 != v111 )
    _libc_free((unsigned __int64)v133);
  if ( v125 )
    sub_B91220((__int64)v19, v125);
  return 1;
}
