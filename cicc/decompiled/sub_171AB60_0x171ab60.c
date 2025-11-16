// Function: sub_171AB60
// Address: 0x171ab60
//
__int64 *__fastcall sub_171AB60(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __m128i a4,
        double a5,
        double a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  _QWORD *v9; // r14
  _OWORD *v10; // rax
  __int64 v11; // rbx
  unsigned int v12; // r8d
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  _QWORD **v19; // rax
  _QWORD *v20; // rbx
  __int64 v21; // rax
  unsigned int v22; // ebx
  _BYTE *v23; // r8
  unsigned int v24; // esi
  int v25; // edi
  __int64 **v26; // r12
  __int64 **v27; // rbx
  __int64 **v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rcx
  unsigned __int16 v31; // dx
  char v32; // r13
  __int64 *v33; // r15
  __int64 v34; // rdx
  __int64 *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r14
  int v40; // eax
  __int64 v41; // rdx
  __int16 v42; // dx
  __int64 v43; // rdx
  _QWORD *v44; // r13
  __int16 *v45; // rax
  __int64 v46; // r14
  char v47; // r15
  void *v48; // r12
  bool v49; // al
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r12
  __int64 v53; // r14
  void *v54; // r12
  _OWORD *v55; // rbx
  __int16 *v57; // r15
  _BYTE *v58; // rax
  __int64 *v60; // rdi
  float v61; // xmm0_4
  __int64 *v62; // rdi
  float v63; // xmm0_4
  _QWORD *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  _QWORD **v68; // rdi
  __int64 v69; // rax
  char *v70; // rax
  char v71; // al
  __int64 *v72; // rdi
  float v73; // xmm0_4
  __int64 *v74; // rdi
  float v75; // xmm0_4
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 i; // r12
  _QWORD *v79; // rax
  _QWORD *v80; // rdx
  _BYTE *v81; // rax
  __int64 *v83; // rdi
  float v84; // xmm0_4
  __int64 *v85; // rdi
  float v86; // xmm0_4
  __int16 *v87; // [rsp+8h] [rbp-228h]
  float v88; // [rsp+18h] [rbp-218h]
  __int16 *v89; // [rsp+18h] [rbp-218h]
  float v90; // [rsp+18h] [rbp-218h]
  float v91; // [rsp+18h] [rbp-218h]
  float v92; // [rsp+18h] [rbp-218h]
  __int64 v93; // [rsp+20h] [rbp-210h]
  _QWORD *v95; // [rsp+38h] [rbp-1F8h]
  __int64 v96; // [rsp+40h] [rbp-1F0h]
  __int64 v98; // [rsp+58h] [rbp-1D8h]
  __int64 v99; // [rsp+60h] [rbp-1D0h]
  _QWORD *v100; // [rsp+68h] [rbp-1C8h]
  unsigned int v101; // [rsp+84h] [rbp-1ACh]
  _QWORD *v102; // [rsp+A0h] [rbp-190h]
  __int64 *v103; // [rsp+A8h] [rbp-188h]
  __int64 v104; // [rsp+B0h] [rbp-180h]
  _OWORD *v105; // [rsp+B8h] [rbp-178h]
  unsigned int v106; // [rsp+C0h] [rbp-170h]
  unsigned int v107; // [rsp+C0h] [rbp-170h]
  unsigned int v108; // [rsp+C0h] [rbp-170h]
  unsigned int v109; // [rsp+C4h] [rbp-16Ch]
  bool v110; // [rsp+C4h] [rbp-16Ch]
  __int64 v111; // [rsp+C8h] [rbp-168h]
  __int64 **v112; // [rsp+C8h] [rbp-168h]
  __int64 *v113; // [rsp+C8h] [rbp-168h]
  char v114; // [rsp+DFh] [rbp-151h] BYREF
  __int64 v115[4]; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v116; // [rsp+100h] [rbp-130h] BYREF
  void *v117; // [rsp+108h] [rbp-128h] BYREF
  __int64 v118; // [rsp+110h] [rbp-120h]
  char v119[8]; // [rsp+120h] [rbp-110h] BYREF
  void *v120[3]; // [rsp+128h] [rbp-108h] BYREF
  _BYTE *v121; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v122; // [rsp+148h] [rbp-E8h]
  _BYTE v123[32]; // [rsp+150h] [rbp-E0h] BYREF
  _QWORD v124[18]; // [rsp+170h] [rbp-C0h] BYREF
  _OWORD v125[3]; // [rsp+200h] [rbp-30h] BYREF

  v9 = (_QWORD *)a2;
  v101 = *(_DWORD *)(a2 + 8);
  v10 = v124;
  do
  {
    *(_QWORD *)v10 = 0;
    v10 += 3;
    *((_BYTE *)v10 - 40) = 0;
    *((_BYTE *)v10 - 39) = 0;
    *((_WORD *)v10 - 19) = 0;
  }
  while ( v125 != v10 );
  v121 = v123;
  v122 = 0x400000000LL;
  if ( !v101 )
  {
LABEL_94:
    v33 = (__int64 *)sub_15A10B0(**(_QWORD **)(a1 + 8), 0.0);
    goto LABEL_95;
  }
  v11 = 1;
  v12 = 0;
  v109 = 0;
  v104 = v101 - 1 + 2LL;
  v100 = 0;
  while ( 1 )
  {
    v13 = 8 * v11;
    v14 = *(__int64 **)(*v9 + 8 * v11 - 8);
    if ( !v14 )
    {
      v106 = v12;
      v111 = v11 + 1;
      goto LABEL_6;
    }
    v15 = *v14;
    v16 = v109;
    if ( HIDWORD(v122) <= v109 )
    {
      v107 = v12;
      v113 = *(__int64 **)(*v9 + 8 * v11 - 8);
      sub_16CD150((__int64)&v121, v123, 0, 8, v12, a9);
      v16 = (unsigned int)v122;
      v12 = v107;
      v14 = v113;
    }
    *(_QWORD *)&v121[8 * v16] = v14;
    v111 = v11 + 1;
    v17 = (unsigned int)(v122 + 1);
    LODWORD(v122) = v122 + 1;
    if ( v101 > (unsigned int)v11 )
    {
      v18 = 8 * (v11 + 1 + v101 - 1 - (unsigned int)v11);
      do
      {
        while ( 1 )
        {
          v19 = (_QWORD **)(v13 + *v9);
          v20 = *v19;
          if ( *v19 )
          {
            if ( v15 == *v20 )
              break;
          }
          v13 += 8;
          if ( v18 == v13 )
            goto LABEL_19;
        }
        *v19 = 0;
        v21 = (unsigned int)v122;
        if ( (unsigned int)v122 >= HIDWORD(v122) )
        {
          v108 = v12;
          sub_16CD150((__int64)&v121, v123, 0, 8, v12, a9);
          v21 = (unsigned int)v122;
          v12 = v108;
        }
        v13 += 8;
        *(_QWORD *)&v121[8 * v21] = v20;
        LODWORD(v122) = v122 + 1;
      }
      while ( v18 != v13 );
LABEL_19:
      v17 = (unsigned int)v122;
    }
    v22 = v109 + 1;
    if ( v109 + 1 == (_DWORD)v17 )
      break;
    v106 = v12 + 1;
    v99 = v12;
    v98 = 6LL * v12;
    v93 = 8LL * v109;
    v41 = *(_QWORD *)&v121[v93];
    v124[v98] = *(_QWORD *)v41;
    v102 = &v124[v98 + 1];
    if ( *(_BYTE *)(v41 + 8) )
    {
      sub_171A6E0((__int64)v102, v41 + 16);
      v17 = (unsigned int)v122;
    }
    else
    {
      v42 = *(_WORD *)(v41 + 10);
      LOBYTE(v124[6 * v12 + 1]) = 0;
      WORD1(v124[6 * v12 + 1]) = v42;
    }
    v43 = v22;
    if ( v22 < (unsigned int)v17 )
    {
      v96 = v15;
      v95 = v9;
      v44 = &v124[v98 + 2];
      v103 = &v124[v98 + 3];
      v105 = &v125[3 * v99];
      while ( 1 )
      {
        while ( 1 )
        {
          v46 = *(_QWORD *)&v121[8 * v43];
          v47 = *((_BYTE *)v105 - 136);
          if ( v47 != *(_BYTE *)(v46 + 8) )
            break;
          if ( !v47 )
          {
            *((_WORD *)v105 - 67) += *(_WORD *)(v46 + 10);
            goto LABEL_62;
          }
          v54 = sub_16982C0();
          if ( (void *)v124[v98 + 3] == v54 )
          {
LABEL_93:
            sub_16A0E00(v103, (__int64 *)(v46 + 24), 0, *(double *)a4.m128i_i64, a5, a6);
            goto LABEL_62;
          }
          if ( (unsigned __int8)sub_169DE70((__int64)v44) || (unsigned __int8)sub_169DE70(v46 + 16) )
          {
            if ( v54 == (void *)v124[v98 + 3] )
              goto LABEL_123;
LABEL_91:
            sub_16986F0(v103, 0, 0, 0);
            goto LABEL_62;
          }
          v57 = (__int16 *)sub_1698270();
          if ( (__int16 *)v124[v98 + 3] == v57 )
          {
            v58 = sub_16D40F0((__int64)qword_4FBB490);
            if ( v58 ? *v58 : LOBYTE(qword_4FBB490[2]) )
            {
              v60 = (__int64 *)(v46 + 24);
              if ( v54 == *(void **)(v46 + 24) )
                v60 = (__int64 *)(*(_QWORD *)(v46 + 32) + 8LL);
              v61 = sub_169D890(v60);
              v62 = &v124[v98 + 3];
              if ( v54 == (void *)*v62 )
                v62 = (__int64 *)(v124[v98 + 4] + 8LL);
              v88 = v61;
              v63 = sub_169D890(v62);
              *(_QWORD *)&a5 = LODWORD(v88);
              a4 = (__m128i)COERCE_UNSIGNED_INT(sub_1C40DF0(v115, 1, 1, v63, v88));
              if ( (unsigned int)sub_1C40EE0(v115) )
              {
LABEL_170:
                sub_169CB40((__int64)v44, 0, 0, 0, *(float *)a4.m128i_i32);
                goto LABEL_62;
              }
              sub_169D3B0((__int64)&v116, a4);
              sub_169E320(v120, &v116, v57);
              sub_1698460((__int64)&v116);
              goto LABEL_166;
            }
          }
LABEL_69:
          sub_169CEB0((__int16 **)v103, (_BYTE *)(v46 + 24), 0);
          v17 = (unsigned int)v122;
          v43 = v22 + 1;
          v22 = v43;
          if ( (unsigned int)v43 >= (unsigned int)v122 )
          {
LABEL_70:
            v15 = v96;
            v9 = v95;
            goto LABEL_71;
          }
        }
        v48 = sub_16982C0();
        if ( !v47 )
        {
          sub_1719070((__int64)v102, *(_QWORD *)(v46 + 24), *(double *)a4.m128i_i64, a5, a6);
          if ( (void *)v124[v98 + 3] == v48 )
            goto LABEL_93;
          if ( !(unsigned __int8)sub_169DE70((__int64)v44) && !(unsigned __int8)sub_169DE70(v46 + 16) )
          {
            if ( (void *)v124[v98 + 3] == sub_1698270() )
            {
              v81 = sub_16D40F0((__int64)qword_4FBB490);
              if ( v81 ? *v81 : LOBYTE(qword_4FBB490[2]) )
              {
                v83 = (__int64 *)(v46 + 24);
                if ( *(void **)(v46 + 24) == v48 )
                  v83 = (__int64 *)(*(_QWORD *)(v46 + 32) + 8LL);
                v84 = sub_169D890(v83);
                v85 = &v124[v98 + 3];
                if ( (void *)*v85 == v48 )
                  v85 = (__int64 *)(v124[v98 + 4] + 8LL);
                v92 = v84;
                v86 = sub_169D890(v85);
                *(_QWORD *)&a5 = LODWORD(v92);
                a4.m128i_i64[0] = COERCE_UNSIGNED_INT(sub_1C40DF0(&v116, 1, 1, v86, v92));
                if ( (unsigned int)sub_1C40EE0(&v116) )
                  goto LABEL_170;
                sub_14D1B70((__int64)v119, 1, *(float *)a4.m128i_i32);
LABEL_166:
                sub_171A510((void **)v103, v120);
                sub_127D120(v120);
                goto LABEL_62;
              }
            }
            goto LABEL_69;
          }
          if ( (void *)v124[v98 + 3] == v48 )
          {
LABEL_123:
            sub_169CAA0((__int64)v103, 0, 0, 0, *(float *)a4.m128i_i32);
            goto LABEL_62;
          }
          goto LABEL_91;
        }
        sub_171A3E0((__int64)&v116, v124[v98 + 3], *(__int16 *)(v46 + 10), *(double *)a4.m128i_i64, a5, a6);
        if ( (void *)v124[v98 + 3] == v48 )
        {
          sub_16A0E00(v103, (__int64 *)&v117, 0, *(double *)a4.m128i_i64, a5, a6);
        }
        else if ( (unsigned __int8)sub_169DE70((__int64)v44) || (unsigned __int8)sub_169DE70((__int64)&v116) )
        {
          if ( (void *)v124[v98 + 3] != v48 )
          {
            sub_16986F0(v103, 0, 0, 0);
            if ( v117 == v48 )
            {
LABEL_83:
              v50 = v118;
              if ( v118 )
              {
                v51 = 32LL * *(_QWORD *)(v118 - 8);
                v52 = v118 + v51;
                if ( v118 != v118 + v51 )
                {
                  v53 = v118;
                  do
                  {
                    v52 -= 32;
                    sub_127D120((_QWORD *)(v52 + 8));
                  }
                  while ( v53 != v52 );
                  v50 = v53;
                }
                j_j_j___libc_free_0_0(v50 - 8);
              }
              goto LABEL_62;
            }
            goto LABEL_61;
          }
          sub_169CAA0((__int64)v103, 0, 0, 0, *(float *)a4.m128i_i32);
        }
        else
        {
          v45 = (__int16 *)sub_1698270();
          if ( (__int16 *)v124[v98 + 3] == v45
            && ((v89 = v45, (v70 = (char *)sub_16D40F0((__int64)qword_4FBB490)) == 0)
              ? (v71 = qword_4FBB490[2])
              : (v71 = *v70),
                v71) )
          {
            v72 = (__int64 *)&v117;
            if ( v117 == v48 )
              v72 = (__int64 *)(v118 + 8);
            v73 = sub_169D890(v72);
            v74 = &v124[v98 + 3];
            if ( (void *)*v74 == v48 )
              v74 = (__int64 *)(v124[v98 + 4] + 8LL);
            v87 = v89;
            v90 = v73;
            v75 = sub_169D890(v74);
            *(_QWORD *)&a5 = LODWORD(v90);
            v91 = sub_1C40DF0(&v114, 1, 1, v75, v90);
            a4.m128i_i64[0] = LODWORD(v91);
            if ( (unsigned int)sub_1C40EE0(&v114) )
            {
              sub_169CB40((__int64)v44, 0, 0, 0, v91);
            }
            else
            {
              sub_169D3B0((__int64)v115, (__m128i)LODWORD(v91));
              sub_169E320(v120, v115, v87);
              sub_1698460((__int64)v115);
              sub_171A510((void **)v103, v120);
              sub_127D120(v120);
            }
          }
          else
          {
            sub_169CEB0((__int16 **)v103, &v117, 0);
          }
        }
        if ( v117 == v48 )
          goto LABEL_83;
LABEL_61:
        sub_1698460((__int64)&v117);
LABEL_62:
        v17 = (unsigned int)v122;
        v43 = v22 + 1;
        v22 = v43;
        if ( (unsigned int)v43 >= (unsigned int)v122 )
          goto LABEL_70;
      }
    }
LABEL_71:
    if ( v109 < v17 )
      goto LABEL_72;
    if ( v109 > v17 )
    {
      if ( v109 > (unsigned __int64)HIDWORD(v122) )
        sub_16CD150((__int64)&v121, v123, v109, 8, v12, a9);
      v79 = &v121[8 * (unsigned int)v122];
      v80 = &v121[v93];
      if ( v79 != (_QWORD *)&v121[v93] )
      {
        do
        {
          if ( v79 )
            *v79 = 0;
          ++v79;
        }
        while ( v80 != v79 );
      }
LABEL_72:
      LODWORD(v122) = v109;
      if ( !v15 )
        goto LABEL_106;
      goto LABEL_73;
    }
    v109 = v122;
    if ( !v15 )
    {
LABEL_106:
      v100 = &v124[v98];
      goto LABEL_6;
    }
LABEL_73:
    if ( LOBYTE(v124[6 * v99 + 1]) )
    {
      v64 = &v124[v98 + 2];
      if ( (void *)v124[v98 + 3] == sub_16982C0() )
        v64 = (_QWORD *)v124[v98 + 4];
      v49 = (*((_BYTE *)v64 + 26) & 7) == 3;
    }
    else
    {
      v49 = WORD1(v124[6 * v99 + 1]) == 0;
    }
    if ( !v49 )
    {
      if ( v109 >= HIDWORD(v122) )
      {
        sub_16CD150((__int64)&v121, v123, 0, 8, v12, a9);
        v109 = v122;
      }
      *(_QWORD *)&v121[8 * v109] = &v124[v98];
      v109 = v122 + 1;
      LODWORD(v122) = v122 + 1;
    }
LABEL_6:
    v11 = v111;
    if ( v104 == v111 )
      goto LABEL_22;
LABEL_7:
    v12 = v106;
  }
  v109 = v17;
  v11 = v111;
  v106 = v12;
  if ( v104 != v111 )
    goto LABEL_7;
LABEL_22:
  if ( v100 )
  {
    if ( v109 >= HIDWORD(v122) )
    {
      sub_16CD150((__int64)&v121, v123, 0, 8, v12, a9);
      v109 = v122;
    }
    *(_QWORD *)&v121[8 * v109] = v100;
    v109 = v122 + 1;
    LODWORD(v122) = v122 + 1;
  }
  if ( !v109 )
    goto LABEL_94;
  v23 = v121;
  v24 = v109 - 1;
  v25 = 0;
  v26 = (__int64 **)&v121[8 * v109];
  v27 = (__int64 **)v121;
  v28 = (__int64 **)v121;
  do
  {
    while ( 1 )
    {
      v29 = *v28;
      v30 = **v28;
      if ( !v30 || *(_BYTE *)(v30 + 16) == 9 )
        goto LABEL_30;
      if ( *((_BYTE *)v29 + 8) )
      {
LABEL_29:
        ++v24;
        goto LABEL_30;
      }
      v31 = *((_WORD *)v29 + 5);
      if ( v31 >= 0xFFFEu )
        break;
      if ( v31 != 1 )
        goto LABEL_29;
LABEL_30:
      if ( v26 == ++v28 )
        goto LABEL_37;
    }
    ++v25;
    if ( v31 != 0xFFFF )
      goto LABEL_29;
    ++v28;
  }
  while ( v26 != v28 );
LABEL_37:
  if ( a3 < (v25 == v109) + v24 )
  {
    v33 = 0;
    goto LABEL_96;
  }
  v112 = (__int64 **)&v121[8 * v109];
  v32 = 0;
  v33 = 0;
  while ( 2 )
  {
    while ( 2 )
    {
      v35 = *v27;
      v39 = **v27;
      v38 = *((unsigned __int8 *)*v27 + 8);
      if ( !v39 )
      {
        v68 = **(_QWORD ****)(a1 + 8);
        if ( (_BYTE)v38 )
        {
          v35 += 2;
          v69 = sub_159CCF0(*v68, (__int64)v35);
        }
        else
        {
          *(double *)a4.m128i_i64 = (double)*((__int16 *)v35 + 5);
          v69 = sub_15A10B0((__int64)v68, *(double *)a4.m128i_i64);
        }
        v38 = 0;
        v39 = v69;
        goto LABEL_41;
      }
      if ( (_BYTE)v38 )
      {
        v34 = sub_159CCF0(***(_QWORD ****)(a1 + 8), (__int64)(v35 + 2));
        goto LABEL_40;
      }
      v40 = *((__int16 *)v35 + 5);
      v37 = (unsigned int)(v40 + 1);
      LOBYTE(v38) = *((_WORD *)v35 + 5) == 0xFFFF;
      if ( ((*((_WORD *)v35 + 5) + 1) & 0xFFFD) != 0 )
      {
        if ( (((_WORD)v40 + 2) & 0xFFFB) != 0 )
        {
          *(double *)a4.m128i_i64 = (double)v40;
          v34 = sub_15A10B0(**(_QWORD **)(a1 + 8), (double)v40);
LABEL_40:
          v35 = (__int64 *)v39;
          v36 = sub_1719390((__int64 *)a1, v39, v34, *(double *)a4.m128i_i64, a5, a6);
          v38 = 0;
          v39 = v36;
          goto LABEL_41;
        }
        v35 = (__int64 *)**v27;
        v110 = (_WORD)v40 == 0xFFFE;
        v67 = sub_1719710((__int64 *)a1, (__int64)v35, (__int64)v35, *(double *)a4.m128i_i64, a5, a6);
        v38 = v110;
        v39 = v67;
        if ( !v33 )
        {
LABEL_130:
          v32 = v38;
          v33 = (__int64 *)v39;
          goto LABEL_45;
        }
      }
      else
      {
LABEL_41:
        if ( !v33 )
          goto LABEL_130;
      }
      if ( v32 == (_BYTE)v38 )
      {
        v35 = v33;
        v33 = (__int64 *)sub_1719710((__int64 *)a1, (__int64)v33, v39, *(double *)a4.m128i_i64, a5, a6);
LABEL_45:
        if ( v112 == ++v27 )
          goto LABEL_126;
        continue;
      }
      break;
    }
    if ( v32 )
    {
      v35 = (__int64 *)v39;
      v32 = 0;
      v33 = (__int64 *)sub_1719550((__int64 *)a1, v39, (__int64)v33, *(double *)a4.m128i_i64, a5, a6);
      goto LABEL_45;
    }
    v35 = v33;
    ++v27;
    v33 = (__int64 *)sub_1719550((__int64 *)a1, (__int64)v33, v39, *(double *)a4.m128i_i64, a5, a6);
    if ( v112 != v27 )
      continue;
    break;
  }
LABEL_126:
  if ( v32 )
  {
    v65 = sub_15A14C0(*v33, (__int64)v35, v37, v38);
    v66 = sub_1719550((__int64 *)a1, v65, (__int64)v33, *(double *)a4.m128i_i64, a5, a6);
    v33 = (__int64 *)v66;
    if ( *(_BYTE *)(v66 + 16) > 0x17u )
      sub_1718FD0(a1, v66);
  }
LABEL_95:
  v23 = v121;
LABEL_96:
  if ( v23 != v123 )
    _libc_free((unsigned __int64)v23);
  v55 = v125;
  do
  {
    v55 -= 3;
    if ( *((_BYTE *)v55 + 9) )
    {
      if ( *((void **)v55 + 3) == sub_16982C0() )
      {
        v76 = *((_QWORD *)v55 + 4);
        if ( v76 )
        {
          v77 = 32LL * *(_QWORD *)(v76 - 8);
          for ( i = v76 + v77; v76 != i; sub_127D120((_QWORD *)(i + 8)) )
            i -= 32;
          j_j_j___libc_free_0_0(v76 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)v55 + 24);
      }
    }
  }
  while ( v55 != (_OWORD *)v124 );
  return v33;
}
