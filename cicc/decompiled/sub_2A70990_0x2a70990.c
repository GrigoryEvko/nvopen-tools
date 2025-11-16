// Function: sub_2A70990
// Address: 0x2a70990
//
void __fastcall sub_2A70990(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rbx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned __int8 v6; // al
  __int64 v7; // r14
  unsigned int v8; // edi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 v12; // r14
  unsigned int i; // r15d
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // r14
  int v23; // edx
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 **v31; // rbx
  unsigned int v32; // eax
  unsigned int v33; // esi
  __int64 v34; // rax
  __m128i *v35; // r14
  const void **v36; // rdx
  int v37; // ecx
  const void **v38; // rax
  __int32 v39; // edx
  __int32 v40; // edx
  unsigned __int8 *v41; // r15
  _QWORD *v42; // rax
  char v43; // r12
  _QWORD *v44; // r14
  const void **v45; // r9
  __int64 v46; // rax
  __int64 *v47; // rbx
  __int64 *v48; // r12
  unsigned __int8 *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 *v54; // rcx
  __int64 v55; // r9
  __int64 v56; // rax
  int v57; // ecx
  __int64 v58; // rsi
  int v59; // ecx
  unsigned int v60; // edx
  __int64 *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rsi
  unsigned __int8 *v64; // rax
  const void **v65; // rax
  __int64 v66; // rdx
  unsigned int v67; // eax
  unsigned int v68; // eax
  unsigned int v69; // eax
  signed __int64 v70; // r14
  unsigned int v71; // r14d
  __int64 v72; // rax
  __int64 *v73; // rax
  __int64 *v74; // rax
  __m128i *v75; // rsi
  int v76; // eax
  int v77; // r9d
  int v78; // ecx
  int v79; // r10d
  unsigned int v80; // eax
  unsigned int v81; // eax
  unsigned int v82; // eax
  int v83; // eax
  int v84; // r8d
  bool v85; // [rsp+8h] [rbp-1B8h]
  unsigned int v86; // [rsp+10h] [rbp-1B0h]
  __int64 *v87; // [rsp+18h] [rbp-1A8h]
  __int64 v89; // [rsp+28h] [rbp-198h]
  int v90; // [rsp+28h] [rbp-198h]
  __int64 v91; // [rsp+30h] [rbp-190h]
  int v92; // [rsp+38h] [rbp-188h]
  unsigned __int8 **v93; // [rsp+38h] [rbp-188h]
  unsigned __int8 *v94; // [rsp+38h] [rbp-188h]
  int v95; // [rsp+40h] [rbp-180h] BYREF
  unsigned __int8 *v96; // [rsp+48h] [rbp-178h]
  char v97; // [rsp+50h] [rbp-170h]
  const void **v98; // [rsp+60h] [rbp-160h] BYREF
  unsigned int v99; // [rsp+68h] [rbp-158h]
  unsigned __int64 v100; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v101; // [rsp+78h] [rbp-148h]
  const void **v102; // [rsp+80h] [rbp-140h] BYREF
  unsigned int v103; // [rsp+88h] [rbp-138h]
  const void **v104; // [rsp+90h] [rbp-130h] BYREF
  unsigned int v105; // [rsp+98h] [rbp-128h]
  const void **v106; // [rsp+A0h] [rbp-120h] BYREF
  unsigned int v107; // [rsp+A8h] [rbp-118h]
  const void **v108; // [rsp+B0h] [rbp-110h] BYREF
  unsigned int v109; // [rsp+B8h] [rbp-108h]
  const void **v110; // [rsp+C0h] [rbp-100h] BYREF
  unsigned int v111; // [rsp+C8h] [rbp-F8h]
  __int64 v112[2]; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v113; // [rsp+E0h] [rbp-E0h] BYREF
  unsigned __int32 v114; // [rsp+E8h] [rbp-D8h]
  unsigned __int64 v115; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned int v116; // [rsp+F8h] [rbp-C8h]
  __m128i v117; // [rsp+110h] [rbp-B0h] BYREF
  unsigned __int64 v118; // [rsp+120h] [rbp-A0h] BYREF
  unsigned int v119; // [rsp+128h] [rbp-98h]
  const void **v120; // [rsp+140h] [rbp-80h] BYREF
  __int64 v121; // [rsp+148h] [rbp-78h]
  const void **v122; // [rsp+150h] [rbp-70h] BYREF
  unsigned int v123; // [rsp+158h] [rbp-68h]

  v4 = (__int64)a2;
  v5 = *((_QWORD *)a2 - 4);
  v6 = *a2;
  if ( !v5 )
    goto LABEL_14;
  if ( !*(_BYTE *)v5 && *((_QWORD *)a2 + 10) == *(_QWORD *)(v5 + 24) )
  {
    if ( v6 != 85 )
    {
      if ( sub_B2FC80(*((_QWORD *)a2 - 4)) )
        goto LABEL_14;
      goto LABEL_17;
    }
    v7 = *((_QWORD *)a2 - 4);
  }
  else
  {
    if ( v6 != 85 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10) )
      goto LABEL_14;
    v7 = 0;
  }
  if ( (*(_BYTE *)(v5 + 33) & 0x20) == 0 )
  {
LABEL_12:
    if ( !v7 )
      goto LABEL_14;
    v5 = v7;
    if ( sub_B2FC80(v7) )
      goto LABEL_14;
LABEL_17:
    v11 = *(__int64 **)(*(_QWORD *)(v5 + 24) + 16LL);
    v12 = *v11;
    if ( *(_BYTE *)(*v11 + 8) == 15 )
    {
      if ( (unsigned __int8)sub_B19060(a1 + 360, v5, v9, v10) )
      {
        v92 = *(_DWORD *)(v12 + 12);
        if ( v92 )
        {
          v89 = v5;
          for ( i = 0; i != v92; ++i )
          {
            LOWORD(v2) = 256;
            v117.m128i_i32[2] = i;
            v2 = ((unsigned __int64)(unsigned int)qword_500BEC8 << 32) | (unsigned int)v2;
            v117.m128i_i64[0] = v89;
            v14 = (unsigned __int8 *)sub_2A6EC30(a1 + 280, &v117);
            sub_22C05A0((__int64)&v120, v14);
            v15 = sub_2A6A1C0(a1, a2, i);
            sub_2A639B0(a1, v15, (__int64)a2, (__int64)&v120, v2);
            sub_22C0090((unsigned __int8 *)&v120);
          }
        }
        return;
      }
    }
    else
    {
      v16 = *(unsigned int *)(a1 + 256);
      v17 = *(_QWORD *)(a1 + 240);
      if ( (_DWORD)v16 )
      {
        v18 = (v16 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v5 != *v19 )
        {
          v76 = 1;
          while ( v20 != -4096 )
          {
            v77 = v76 + 1;
            v18 = (v16 - 1) & (v76 + v18);
            v19 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v19;
            if ( v5 == *v19 )
              goto LABEL_25;
            v76 = v77;
          }
          goto LABEL_14;
        }
LABEL_25:
        if ( v19 != (__int64 *)(v17 + 16 * v16) )
        {
          v21 = *(_QWORD *)(a1 + 264) + 48LL * *((unsigned int *)v19 + 2);
          if ( v21 != *(_QWORD *)(a1 + 264) + 48LL * *(unsigned int *)(a1 + 272) )
          {
            v22 = (unsigned int)qword_500BEC8;
            sub_22C05A0((__int64)&v120, (unsigned __int8 *)(v21 + 8));
            sub_2A689D0(a1, v4, (unsigned __int8 *)&v120, (v22 << 32) | 0x100);
            sub_22C0090((unsigned __int8 *)&v120);
            return;
          }
        }
      }
    }
LABEL_14:
    sub_2A6B3D0(a1, (unsigned __int8 *)v4);
    return;
  }
  v8 = *(_DWORD *)(v5 + 36);
  if ( v8 == 336 )
  {
    v120 = (const void **)a2;
    if ( *(_BYTE *)sub_2A686D0(a1 + 136, (__int64 *)&v120) == 6 )
      return;
    v91 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v49 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)v91);
    sub_22C05A0((__int64)&v113, v49);
    v50 = *(_QWORD *)(a1 + 2544);
    v51 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 72LL);
    v52 = *(unsigned int *)(a1 + 2560);
    if ( (_DWORD)v52 )
    {
      v53 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v54 = (__int64 *)(v50 + 16LL * v53);
      v55 = *v54;
      if ( v51 == *v54 )
      {
LABEL_72:
        if ( v54 != (__int64 *)(v50 + 16 * v52) )
        {
          v56 = v54[1];
          v57 = *(_DWORD *)(v56 + 48);
          v58 = *(_QWORD *)(v56 + 32);
          if ( v57 )
          {
            v59 = v57 - 1;
            v60 = v59 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v61 = (__int64 *)(v58 + 16LL * v60);
            v62 = *v61;
            if ( v4 == *v61 )
            {
LABEL_75:
              v63 = v61[1];
              goto LABEL_76;
            }
            v83 = 1;
            while ( v62 != -4096 )
            {
              v84 = v83 + 1;
              v60 = v59 & (v83 + v60);
              v61 = (__int64 *)(v58 + 16LL * v60);
              v62 = *v61;
              if ( *v61 == v4 )
                goto LABEL_75;
              v83 = v84;
            }
          }
        }
      }
      else
      {
        v78 = 1;
        while ( v55 != -4096 )
        {
          v79 = v78 + 1;
          v53 = (v52 - 1) & (v78 + v53);
          v54 = (__int64 *)(v50 + 16LL * v53);
          v55 = *v54;
          if ( v51 == *v54 )
            goto LABEL_72;
          v78 = v79;
        }
      }
    }
    v63 = 0;
LABEL_76:
    sub_2A457F0((__int64)&v95, v63);
    if ( !v97 )
    {
      sub_22C05A0((__int64)&v120, (unsigned __int8 *)&v113);
      v117.m128i_i64[0] = v4;
      v74 = sub_2A686D0(a1 + 136, v117.m128i_i64);
      sub_2A639B0(a1, v74, v4, (__int64)&v120, 0x100000000LL);
      sub_22C0090((unsigned __int8 *)&v120);
      goto LABEL_91;
    }
    v90 = v95;
    v94 = v96;
    if ( !*(_BYTE *)sub_2A68BC0(a1, v96) )
    {
      sub_2A6FB70(a1, (__int64)v94, v4);
      goto LABEL_91;
    }
    v64 = (unsigned __int8 *)sub_2A68BC0(a1, v94);
    sub_22C05A0((__int64)&v117, v64);
    v120 = (const void **)v4;
    v87 = sub_2A686D0(a1 + 136, (__int64 *)&v120);
    if ( (unsigned __int8)(v117.m128i_i8[0] - 4) <= 1u || (unsigned __int8)(v113 - 4) <= 1u )
    {
      v65 = (const void **)sub_9208B0(*(_QWORD *)a1, *(_QWORD *)(v91 + 8));
      v121 = v66;
      v120 = v65;
      v67 = sub_CA1930(&v120);
      sub_AADB10((__int64)&v98, v67, 1);
      if ( (unsigned __int8)(v117.m128i_i8[0] - 4) <= 1u )
      {
        sub_AB15A0((__int64)&v120, v90, (__int64)&v117.m128i_i64[1]);
        if ( v99 > 0x40 && v98 )
          j_j___libc_free_0_0((unsigned __int64)v98);
        v98 = v120;
        v68 = v121;
        LODWORD(v121) = 0;
        v99 = v68;
        if ( v101 > 0x40 && v100 )
          j_j___libc_free_0_0(v100);
        v100 = (unsigned __int64)v122;
        v69 = v123;
        v123 = 0;
        v101 = v69;
        sub_969240((__int64 *)&v122);
        sub_969240((__int64 *)&v120);
      }
      sub_2A62360((__int64)&v102, (char *)&v113, *(_QWORD *)(v91 + 8), 1);
      if ( sub_AAF7D0((__int64)&v102) )
      {
        sub_AADB10((__int64)&v120, v103, 1);
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0((unsigned __int64)v102);
        v102 = v120;
        v81 = v121;
        LODWORD(v121) = 0;
        v103 = v81;
        if ( v105 > 0x40 && v104 )
          j_j___libc_free_0_0((unsigned __int64)v104);
        v104 = v122;
        v82 = v123;
        v123 = 0;
        v105 = v82;
        sub_969240((__int64 *)&v122);
        sub_969240((__int64 *)&v120);
      }
      sub_AB2160((__int64)&v106, (__int64)&v98, (__int64)&v102, 0);
      if ( !(unsigned __int8)sub_AB1BB0((__int64)&v102, (__int64)&v106) )
      {
        v111 = v105;
        if ( v105 > 0x40 )
          sub_C43780((__int64)&v110, (const void **)&v104);
        else
          v110 = v104;
        sub_C46A40((__int64)&v110, 1);
        v80 = v111;
        v111 = 0;
        LODWORD(v121) = v80;
        v120 = v110;
        v85 = v103 <= 0x40 ? v102 == v110 : sub_C43C50((__int64)&v102, (const void **)&v120);
        sub_969240((__int64 *)&v120);
        sub_969240((__int64 *)&v110);
        if ( v85 )
        {
          if ( v107 <= 0x40 && v103 <= 0x40 )
          {
            v107 = v103;
            v106 = v102;
          }
          else
          {
            sub_C43990((__int64)&v106, (__int64)&v102);
          }
          if ( v109 <= 0x40 && v105 <= 0x40 )
          {
            v109 = v105;
            v108 = v104;
          }
          else
          {
            sub_C43990((__int64)&v108, (__int64)&v104);
          }
        }
      }
      sub_2A6FB70(a1, (__int64)v94, v4);
      sub_AAF450((__int64)&v110, (__int64)&v106);
      sub_22C06B0((__int64)&v120, (__int64)&v110, 0);
      sub_2A639B0(a1, v87, v4, (__int64)&v120, 0x100000000LL);
      sub_22C0090((unsigned __int8 *)&v120);
      sub_969240(v112);
      sub_969240((__int64 *)&v110);
      sub_969240((__int64 *)&v108);
      sub_969240((__int64 *)&v106);
      sub_969240((__int64 *)&v104);
      sub_969240((__int64 *)&v102);
      sub_969240((__int64 *)&v100);
      sub_969240((__int64 *)&v98);
      goto LABEL_90;
    }
    if ( v90 == 32 )
    {
      if ( (unsigned __int8)(v117.m128i_i8[0] - 2) <= 1u )
      {
        sub_2A6FB70(a1, (__int64)v94, v4);
        v75 = &v117;
        goto LABEL_119;
      }
    }
    else if ( v117.m128i_i8[0] == 2 && v90 == 33 )
    {
      sub_2A6FB70(a1, (__int64)v94, v4);
      LOWORD(v120) = 0;
      sub_2A62A00((__int64)&v120, (unsigned __int8 *)v117.m128i_i64[1]);
      sub_2A639B0(a1, v87, v4, (__int64)&v120, 0x100000000LL);
      sub_22C0090((unsigned __int8 *)&v120);
      goto LABEL_90;
    }
    v75 = (__m128i *)&v113;
LABEL_119:
    sub_22C05A0((__int64)&v120, (unsigned __int8 *)v75);
    sub_2A639B0(a1, v87, v4, (__int64)&v120, 0x100000000LL);
    sub_22C0090((unsigned __int8 *)&v120);
LABEL_90:
    sub_22C0090((unsigned __int8 *)&v117);
LABEL_91:
    sub_22C0090((unsigned __int8 *)&v113);
    return;
  }
  if ( v8 == 493 )
  {
    v71 = sub_BCB060(*((_QWORD *)a2 + 1));
    v72 = sub_B43CB0((__int64)a2);
    sub_988CD0((__int64)&v113, v72, v71);
    v117.m128i_i32[2] = v114;
    if ( v114 > 0x40 )
      sub_C43780((__int64)&v117, (const void **)&v113);
    else
      v117.m128i_i64[0] = v113;
    v119 = v116;
    if ( v116 > 0x40 )
      sub_C43780((__int64)&v118, (const void **)&v115);
    else
      v118 = v115;
    sub_22C06B0((__int64)&v120, (__int64)&v117, 0);
    sub_2A689D0(a1, (__int64)a2, (unsigned __int8 *)&v120, 0x100000000LL);
    sub_22C0090((unsigned __int8 *)&v120);
    sub_969240((__int64 *)&v118);
    sub_969240(v117.m128i_i64);
    sub_969240((__int64 *)&v115);
    sub_969240(&v113);
  }
  else
  {
    if ( !sub_AB4EB0(v8) )
      goto LABEL_12;
    v23 = *a2;
    v120 = (const void **)&v122;
    v121 = 0x200000000LL;
    if ( v23 == 40 )
    {
      v24 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v24 = -32;
      if ( v23 != 85 )
      {
        v24 = -96;
        if ( v23 != 34 )
LABEL_160:
          BUG();
      }
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v25 = sub_BD2BC0((__int64)a2);
      v27 = v25 + v26;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v27 >> 4) )
          goto LABEL_160;
      }
      else if ( (unsigned int)((v27 - sub_BD2BC0((__int64)a2)) >> 4) )
      {
        if ( (a2[7] & 0x80u) == 0 )
          goto LABEL_160;
        v28 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v29 = sub_BD2BC0((__int64)a2);
        v24 -= 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
      }
    }
    v93 = (unsigned __int8 **)&a2[v24];
    v31 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( v31 != (unsigned __int8 **)&a2[v24] )
    {
      while ( 1 )
      {
        v41 = *v31;
        v42 = sub_2A68BC0(a1, *v31);
        v43 = *(_BYTE *)v42;
        v44 = v42;
        if ( *(_BYTE *)v42 <= 1u )
          goto LABEL_64;
        v45 = (const void **)(v42 + 1);
        if ( v43 == 4 )
        {
LABEL_54:
          v117.m128i_i32[2] = *((_DWORD *)v44 + 4);
          if ( v117.m128i_i32[2] > 0x40u )
            sub_C43780((__int64)&v117, v45);
          else
            v117.m128i_i64[0] = v44[1];
          v119 = *((_DWORD *)v44 + 8);
          if ( v119 > 0x40 )
            sub_C43780((__int64)&v118, (const void **)v44 + 3);
          else
            v118 = v44[3];
          goto LABEL_45;
        }
        v32 = sub_BCB060(*((_QWORD *)v41 + 1));
        v33 = v32;
        if ( v43 == 5 )
        {
          v86 = v32;
          v73 = sub_9876C0(v44 + 1);
          v45 = (const void **)(v44 + 1);
          v33 = v86;
          if ( v73 )
            goto LABEL_54;
          if ( *(_BYTE *)v44 == 2 )
          {
LABEL_58:
            sub_AD8380((__int64)&v117, v44[1]);
            goto LABEL_45;
          }
          if ( !*(_BYTE *)v44 )
          {
            sub_AADB10((__int64)&v117, v86, 0);
            goto LABEL_45;
          }
        }
        else if ( v43 == 2 )
        {
          goto LABEL_58;
        }
        sub_AADB10((__int64)&v117, v33, 1);
LABEL_45:
        v34 = (unsigned int)v121;
        v35 = &v117;
        v36 = v120;
        v37 = v121;
        if ( (unsigned __int64)(unsigned int)v121 + 1 > HIDWORD(v121) )
        {
          if ( v120 > (const void **)&v117 || &v117 >= (__m128i *)&v120[4 * (unsigned int)v121] )
          {
            v35 = &v117;
            sub_9D5330((__int64)&v120, (unsigned int)v121 + 1LL);
            v34 = (unsigned int)v121;
            v36 = v120;
            v37 = v121;
          }
          else
          {
            v70 = (char *)&v117 - (char *)v120;
            sub_9D5330((__int64)&v120, (unsigned int)v121 + 1LL);
            v36 = v120;
            v34 = (unsigned int)v121;
            v35 = (__m128i *)((char *)v120 + v70);
            v37 = v121;
          }
        }
        v38 = &v36[4 * v34];
        if ( v38 )
        {
          v39 = v35->m128i_i32[2];
          v35->m128i_i32[2] = 0;
          *((_DWORD *)v38 + 2) = v39;
          *v38 = (const void *)v35->m128i_i64[0];
          v40 = v35[1].m128i_i32[2];
          v35[1].m128i_i32[2] = 0;
          *((_DWORD *)v38 + 6) = v40;
          v38[2] = (const void *)v35[1].m128i_i64[0];
          v37 = v121;
        }
        LODWORD(v121) = v37 + 1;
        if ( v119 > 0x40 && v118 )
          j_j___libc_free_0_0(v118);
        v31 += 4;
        sub_969240(v117.m128i_i64);
        if ( v93 == v31 )
        {
          v4 = (__int64)a2;
          break;
        }
      }
    }
    v46 = *(_QWORD *)(v4 - 32);
    if ( !v46 || *(_BYTE *)v46 || *(_QWORD *)(v46 + 24) != *(_QWORD *)(v4 + 80) )
      BUG();
    sub_ABD750((__int64)&v110, *(_DWORD *)(v46 + 36), (__int64)v120);
    sub_AAF450((__int64)&v113, (__int64)&v110);
    sub_22C06B0((__int64)&v117, (__int64)&v113, 0);
    sub_2A689D0(a1, v4, (unsigned __int8 *)&v117, 0x100000000LL);
    sub_22C0090((unsigned __int8 *)&v117);
    sub_969240((__int64 *)&v115);
    sub_969240(&v113);
    sub_969240(v112);
    sub_969240((__int64 *)&v110);
LABEL_64:
    v47 = (__int64 *)v120;
    v48 = (__int64 *)&v120[4 * (unsigned int)v121];
    if ( v120 != (const void **)v48 )
    {
      do
      {
        v48 -= 4;
        sub_969240(v48 + 2);
        sub_969240(v48);
      }
      while ( v47 != v48 );
      v48 = (__int64 *)v120;
    }
    if ( v48 != (__int64 *)&v122 )
      _libc_free((unsigned __int64)v48);
  }
}
