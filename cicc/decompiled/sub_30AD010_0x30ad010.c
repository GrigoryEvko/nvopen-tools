// Function: sub_30AD010
// Address: 0x30ad010
//
__int64 __fastcall sub_30AD010(__int64 *a1)
{
  _QWORD *v1; // r11
  __int64 v2; // r9
  __int64 v3; // rcx
  int v4; // r15d
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v10; // rsi
  __int64 v11; // rbx
  unsigned __int64 v12; // r13
  __int64 v13; // r10
  __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // r9
  int v17; // edx
  int v18; // eax
  int v19; // edx
  int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rcx
  _BYTE *v23; // rdi
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rdx
  _QWORD *v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 *v30; // rdx
  unsigned __int64 *v31; // rbx
  int v32; // r10d
  unsigned __int64 *v33; // rdx
  __int64 v34; // r9
  __int64 v35; // r11
  _BYTE *v36; // r15
  _BYTE *v37; // r14
  __int64 v38; // rsi
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // r8
  unsigned int v44; // r13d
  unsigned int v45; // r12d
  __int64 v46; // rdx
  signed __int64 v47; // rax
  __int64 v48; // rbx
  unsigned __int16 v49; // si
  __int64 v50; // rdx
  int v51; // ecx
  _BYTE *v52; // rdx
  __int64 *v53; // rdx
  __int64 v54; // rcx
  unsigned __int16 v55; // di
  _BYTE *v56; // rdi
  unsigned int v57; // r13d
  unsigned __int64 *v58; // r12
  __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 *v62; // rdx
  __int64 v63; // rdx
  unsigned __int64 v64; // rcx
  unsigned __int64 v65; // rdi
  char **v66; // rsi
  __int64 v67; // r8
  int v68; // eax
  __int64 v69; // rdx
  _QWORD *v70; // rdi
  unsigned int v71; // eax
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // r10
  unsigned __int64 v74; // rcx
  int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rdx
  __m128i *v78; // rsi
  __m128i v79; // xmm0
  int v80; // edi
  const void *v81; // rsi
  const void *v82; // rsi
  unsigned __int64 v83; // r13
  char *v84; // rbx
  unsigned __int64 v85; // rt0
  bool v86; // of
  __int64 v87; // rdx
  unsigned __int64 v88; // rbx
  signed __int64 v89; // [rsp+8h] [rbp-388h]
  __int64 v90; // [rsp+10h] [rbp-380h]
  int v91; // [rsp+10h] [rbp-380h]
  int v92; // [rsp+10h] [rbp-380h]
  int v93; // [rsp+18h] [rbp-378h]
  int v94; // [rsp+18h] [rbp-378h]
  __int64 v95; // [rsp+18h] [rbp-378h]
  __int64 v96; // [rsp+18h] [rbp-378h]
  __int64 v97; // [rsp+18h] [rbp-378h]
  __int64 v98; // [rsp+20h] [rbp-370h]
  __int64 v99; // [rsp+20h] [rbp-370h]
  unsigned __int64 v100; // [rsp+20h] [rbp-370h]
  __int64 v101; // [rsp+20h] [rbp-370h]
  __int64 v102; // [rsp+20h] [rbp-370h]
  __int64 *v103; // [rsp+30h] [rbp-360h]
  __int64 v104; // [rsp+38h] [rbp-358h]
  __int64 v105; // [rsp+40h] [rbp-350h]
  int v106; // [rsp+50h] [rbp-340h]
  unsigned int v107; // [rsp+54h] [rbp-33Ch]
  unsigned int v108; // [rsp+54h] [rbp-33Ch]
  unsigned int v109; // [rsp+54h] [rbp-33Ch]
  __int64 v110; // [rsp+58h] [rbp-338h]
  __int64 v111; // [rsp+58h] [rbp-338h]
  int v112; // [rsp+58h] [rbp-338h]
  int v113; // [rsp+58h] [rbp-338h]
  int v114; // [rsp+60h] [rbp-330h]
  __int64 v115; // [rsp+60h] [rbp-330h]
  unsigned int v116; // [rsp+60h] [rbp-330h]
  unsigned int v117; // [rsp+60h] [rbp-330h]
  unsigned int v118; // [rsp+68h] [rbp-328h]
  __int64 v119; // [rsp+68h] [rbp-328h]
  __int64 v120; // [rsp+68h] [rbp-328h]
  int v121; // [rsp+68h] [rbp-328h]
  unsigned int v122; // [rsp+68h] [rbp-328h]
  __int64 v123; // [rsp+68h] [rbp-328h]
  __int64 v124; // [rsp+68h] [rbp-328h]
  unsigned int v125; // [rsp+68h] [rbp-328h]
  _BYTE *v126; // [rsp+80h] [rbp-310h] BYREF
  __int64 v127; // [rsp+88h] [rbp-308h]
  _BYTE v128[128]; // [rsp+90h] [rbp-300h] BYREF
  _BYTE *v129; // [rsp+110h] [rbp-280h] BYREF
  __int64 v130; // [rsp+118h] [rbp-278h]
  _BYTE v131[624]; // [rsp+120h] [rbp-270h] BYREF

  v1 = a1;
  v2 = *((unsigned int *)a1 + 4);
  v3 = *a1;
  v129 = v131;
  v130 = 0x400000000LL;
  if ( !(_DWORD)v2 )
  {
    v31 = (unsigned __int64 *)v131;
    v58 = (unsigned __int64 *)v131;
LABEL_99:
    v57 = 1;
    *v1 = v3 - 1;
    goto LABEL_47;
  }
  v4 = v3 - 1;
  v5 = v2;
  v6 = 0;
  v7 = 0;
  v8 = (unsigned int)(v3 - 1);
  do
  {
    while ( 1 )
    {
      v10 = a1[1];
      v11 = v10 + 144 * v7;
      v12 = *(unsigned int *)(v11 + 8);
      if ( *(_DWORD *)(v11 + 8) )
        break;
LABEL_5:
      ++v6;
LABEL_6:
      v7 = v6;
      v2 = v5;
      if ( v5 <= v6 )
        goto LABEL_20;
    }
    v13 = *(_QWORD *)v11;
    v14 = *(_QWORD *)v11 + 16 * v12 - 16;
    v15 = *(unsigned __int16 *)(v14 + 8);
    if ( (_WORD)v15 != (_WORD)v4 || !*(_QWORD *)v14 )
    {
      if ( v15 == (_DWORD)v8 )
      {
        *(_DWORD *)(v11 + 8) = v12 - 1;
        v5 = *((_DWORD *)a1 + 4);
      }
      goto LABEL_5;
    }
    v16 = v10 + 144 * v2 - 144;
    if ( v11 != v16 )
    {
      if ( v13 == v11 + 16 || *(_QWORD *)v16 == v16 + 16 )
      {
        v72 = *(unsigned int *)(v16 + 8);
        if ( v72 > *(unsigned int *)(v11 + 12) )
        {
          v117 = v8;
          v124 = v16;
          sub_C8D5F0(v11, (const void *)(v11 + 16), v72, 0x10u, v8, v16);
          v12 = *(unsigned int *)(v11 + 8);
          v8 = v117;
          v16 = v124;
        }
        if ( *(unsigned int *)(v16 + 12) < v12 )
        {
          v116 = v8;
          v123 = v16;
          sub_C8D5F0(v16, (const void *)(v16 + 16), v12, 0x10u, v8, v16);
          v12 = *(unsigned int *)(v11 + 8);
          v8 = v116;
          v16 = v123;
        }
        v73 = *(unsigned int *)(v16 + 8);
        v74 = v12;
        v75 = *(_DWORD *)(v16 + 8);
        if ( v73 <= v12 )
          v74 = *(unsigned int *)(v16 + 8);
        if ( v74 )
        {
          v76 = 0;
          do
          {
            v77 = v76 + *(_QWORD *)v16;
            v78 = (__m128i *)(v76 + *(_QWORD *)v11);
            v76 += 16;
            v79 = _mm_loadu_si128(v78);
            v78->m128i_i64[0] = *(_QWORD *)v77;
            v78->m128i_i16[4] = *(_WORD *)(v77 + 8);
            *(_QWORD *)v77 = v79.m128i_i64[0];
            *(_WORD *)(v77 + 8) = v79.m128i_i16[4];
          }
          while ( 16 * v74 != v76 );
          v73 = *(unsigned int *)(v16 + 8);
          v12 = *(unsigned int *)(v11 + 8);
          v75 = *(_DWORD *)(v16 + 8);
        }
        if ( v12 <= v73 )
        {
          if ( v12 < v73 )
          {
            v121 = v73;
            v82 = (const void *)(*(_QWORD *)v16 + 16 * v74);
            if ( v82 != (const void *)(16 * v73 + *(_QWORD *)v16) )
            {
              v109 = v8;
              v113 = v74;
              v115 = v16;
              memcpy((void *)(*(_QWORD *)v11 + 16 * v12), v82, 16 * v73 - 16 * v74);
              v8 = v109;
              LODWORD(v74) = v113;
              v16 = v115;
              v75 = *(_DWORD *)(v11 + 8) + v121 - v12;
            }
            *(_DWORD *)(v11 + 8) = v75;
            *(_DWORD *)(v16 + 8) = v74;
          }
          v10 = a1[1];
        }
        else
        {
          v80 = v73;
          v81 = (const void *)(*(_QWORD *)v11 + 16 * v74);
          if ( v81 != (const void *)(16 * v12 + *(_QWORD *)v11) )
          {
            v108 = v8;
            v112 = v74;
            v114 = v73;
            v120 = v16;
            memcpy((void *)(*(_QWORD *)v16 + 16 * v73), v81, 16 * v12 - 16 * v74);
            v16 = v120;
            v8 = v108;
            LODWORD(v74) = v112;
            LODWORD(v73) = v114;
            v80 = *(_DWORD *)(v120 + 8);
          }
          *(_DWORD *)(v16 + 8) = v12 - v73 + v80;
          *(_DWORD *)(v11 + 8) = v74;
          v10 = a1[1];
        }
      }
      else
      {
        *(_QWORD *)v11 = *(_QWORD *)v16;
        v17 = *(_DWORD *)(v16 + 8);
        *(_QWORD *)v16 = v13;
        v18 = *(_DWORD *)(v11 + 8);
        *(_DWORD *)(v11 + 8) = v17;
        v19 = *(_DWORD *)(v16 + 12);
        *(_DWORD *)(v16 + 8) = v18;
        v20 = *(_DWORD *)(v11 + 12);
        *(_DWORD *)(v11 + 12) = v19;
        *(_DWORD *)(v16 + 12) = v20;
        v10 = a1[1];
      }
    }
    v21 = (unsigned int)v130;
    v22 = HIDWORD(v130);
    v23 = v129;
    v24 = v10 + 144LL * *((unsigned int *)a1 + 4) - 144;
    v25 = (unsigned int)v130 + 1LL;
    v26 = v130;
    if ( v25 > HIDWORD(v130) )
    {
      if ( (unsigned __int64)v129 > v24 )
      {
        v125 = v8;
      }
      else
      {
        v125 = v8;
        if ( v24 < (unsigned __int64)&v129[144 * (unsigned int)v130] )
        {
          v83 = v24 - (_QWORD)v129;
          sub_2740590((__int64)&v129, v25, (unsigned int)v130, HIDWORD(v130), v8, v16);
          v23 = v129;
          v21 = (unsigned int)v130;
          v8 = v125;
          v24 = (unsigned __int64)&v129[v83];
          v26 = v130;
          goto LABEL_15;
        }
      }
      sub_2740590((__int64)&v129, v25, (unsigned int)v130, HIDWORD(v130), v8, v16);
      v21 = (unsigned int)v130;
      v23 = v129;
      v8 = v125;
      v26 = v130;
    }
LABEL_15:
    v27 = 144 * v21;
    v28 = &v23[v27];
    if ( v28 )
    {
      *v28 = v28 + 2;
      v28[1] = 0x800000000LL;
      if ( *(_DWORD *)(v24 + 8) )
      {
        v122 = v8;
        sub_30ACEB0((__int64)v28, (char **)v24, v27, v22, v8, v16);
        v26 = v130;
        v8 = v122;
      }
      else
      {
        v26 = v130;
      }
    }
    LODWORD(v130) = v26 + 1;
    v29 = (unsigned int)(*((_DWORD *)a1 + 4) - 1);
    *((_DWORD *)a1 + 4) = v29;
    v5 = v29;
    v30 = (unsigned __int64 *)(a1[1] + 144 * v29);
    if ( (unsigned __int64 *)*v30 == v30 + 2 )
      goto LABEL_6;
    v118 = v8;
    _libc_free(*v30);
    v5 = *((_DWORD *)a1 + 4);
    v8 = v118;
    v7 = v6;
    v2 = v5;
  }
  while ( v5 > v6 );
LABEL_20:
  v31 = (unsigned __int64 *)v129;
  v32 = v4;
  v1 = a1;
  v33 = (unsigned __int64 *)v129;
  if ( !(_DWORD)v130 )
  {
    v3 = *a1;
    v58 = (unsigned __int64 *)v129;
    goto LABEL_99;
  }
  v103 = a1;
  v107 = v130 - 2;
  v119 = 1;
  v104 = 144;
LABEL_22:
  if ( v107 == -1 )
  {
    v1 = v103;
    v3 = *v103;
    v58 = &v33[18 * (unsigned int)v130];
    goto LABEL_99;
  }
  ++v119;
  v34 = v104;
  v35 = v104 - 144;
  while ( 2 )
  {
    v36 = (char *)v33 + v34;
    v37 = (char *)v33 + v35;
    v38 = *(unsigned __int64 *)((char *)v33 + v35);
    v39 = *(unsigned int *)((char *)v33 + v35 + 8);
    if ( !*(_DWORD *)((char *)v33 + v34 + 8)
      || (v40 = *(_QWORD *)v36 + 16LL * *(unsigned int *)((char *)v33 + v34 + 8) - 16, *(_WORD *)(v40 + 8) != (_WORD)v32) )
    {
      if ( *(_DWORD *)((char *)v33 + v35 + 8) && (v41 = v38 + 16 * v39 - 16, *(_WORD *)(v41 + 8) == (_WORD)v32) )
      {
        v42 = *(_QWORD *)v41;
        v43 = 0;
      }
      else
      {
        v42 = 0;
        v43 = 0;
      }
      goto LABEL_29;
    }
    v42 = *(_QWORD *)v40;
    if ( *(_DWORD *)((char *)v33 + v35 + 8) )
    {
      v60 = v38 + 16 * v39 - 16;
      if ( *(_WORD *)(v60 + 8) == (_WORD)v32 )
      {
        v43 = *(_QWORD *)v60;
        if ( *(__int64 *)v60 < 0 )
        {
          if ( v42 >= 0 )
            goto LABEL_112;
        }
        else if ( *(__int64 *)v60 <= 0 || v42 <= 0 )
        {
          goto LABEL_59;
        }
LABEL_75:
        v34 += 144;
        if ( 144 * (v119 + v107) != v34 )
          continue;
        v104 += 144;
        --v107;
        goto LABEL_22;
      }
    }
    break;
  }
  v43 = 0;
LABEL_59:
  if ( v42 >= 0 )
  {
LABEL_112:
    v85 = v42;
    v42 = v43;
    v43 = v85;
  }
  else
  {
    v36 = (char *)v33 + v35;
    v37 = (char *)v33 + v34;
  }
LABEL_29:
  v126 = v128;
  v127 = 0x800000000LL;
  if ( !*((_DWORD *)v36 + 2) )
  {
LABEL_74:
    v31 = v33;
    goto LABEL_75;
  }
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = -v42;
  while ( *((_DWORD *)v37 + 2) > v44 )
  {
    v53 = (__int64 *)(*(_QWORD *)v36 + 16 * v46);
    v54 = *(_QWORD *)v37 + 16LL * v44;
    v49 = *((_WORD *)v53 + 4);
    v55 = *(_WORD *)(v54 + 8);
    if ( v49 >= v55 )
    {
      if ( v49 == v55 )
      {
        ++v45;
        v86 = (unsigned __int128)(*v53 * (__int128)v47) >> 64 != 0;
        v87 = *v53 * v47;
        if ( v86
          || (++v44, v88 = *(_QWORD *)v54 * v43, !is_mul_ok(*(_QWORD *)v54, v43))
          || (v86 = __OFADD__(v88, v87), v48 = v88 + v87, v86) )
        {
LABEL_43:
          v56 = v126;
          goto LABEL_44;
        }
      }
      else
      {
        ++v44;
        v48 = *(_QWORD *)v54 * v43;
        if ( !is_mul_ok(*(_QWORD *)v54, v43) )
          goto LABEL_43;
        v49 = *(_WORD *)(v54 + 8);
      }
    }
    else
    {
      ++v45;
      v48 = *v53 * v47;
      if ( !is_mul_ok(*v53, v47) )
        goto LABEL_43;
    }
    if ( v48 )
    {
      v50 = (unsigned int)v127;
      v51 = v127;
      if ( (unsigned int)v127 < (unsigned __int64)HIDWORD(v127) )
      {
        v52 = &v126[16 * (unsigned int)v127];
        if ( v52 )
        {
          *(_QWORD *)v52 = v48;
          *((_WORD *)v52 + 4) = v49;
          v51 = v127;
        }
        LODWORD(v127) = v51 + 1;
        goto LABEL_39;
      }
      v61 = v105;
      LOWORD(v61) = v49;
      v105 = v61;
      if ( HIDWORD(v127) < (unsigned __int64)(unsigned int)v127 + 1 )
      {
        v89 = v47;
        v106 = v32;
        v90 = v34;
        v95 = v35;
        v100 = v43;
        sub_C8D5F0((__int64)&v126, v128, (unsigned int)v127 + 1LL, 0x10u, v43, v34);
        v50 = (unsigned int)v127;
        v47 = v89;
        v32 = v106;
        v34 = v90;
        v35 = v95;
        v43 = v100;
      }
      v62 = (__int64 *)&v126[16 * v50];
      *v62 = v48;
      v62[1] = v105;
      v46 = v45;
      LODWORD(v127) = v127 + 1;
      if ( *((_DWORD *)v36 + 2) <= v45 )
        break;
    }
    else
    {
LABEL_39:
      v46 = v45;
      if ( *((_DWORD *)v36 + 2) <= v45 )
        break;
    }
  }
  if ( !(_DWORD)v127 )
  {
    if ( v126 == v128 )
    {
LABEL_73:
      v33 = (unsigned __int64 *)v129;
      goto LABEL_74;
    }
    v94 = v32;
    v99 = v34;
    v111 = v35;
    _libc_free((unsigned __int64)v126);
    v33 = (unsigned __int64 *)v129;
    v32 = v94;
    v34 = v99;
    v35 = v111;
    goto LABEL_74;
  }
  v63 = *((unsigned int *)v103 + 4);
  v64 = *((unsigned int *)v103 + 5);
  v65 = v103[1];
  v66 = &v126;
  v67 = v63 + 1;
  v68 = *((_DWORD *)v103 + 4);
  if ( v63 + 1 > v64 )
  {
    v92 = v32;
    v97 = v34;
    v102 = v35;
    if ( v65 > (unsigned __int64)&v126 || (unsigned __int64)&v126 >= v65 + 144 * v63 )
    {
      sub_2740590((__int64)(v103 + 1), v63 + 1, v63, v64, v67, v34);
      v66 = &v126;
      v32 = v92;
      v34 = v97;
      v63 = *((unsigned int *)v103 + 4);
      v65 = v103[1];
      v35 = v102;
      v68 = *((_DWORD *)v103 + 4);
    }
    else
    {
      v84 = (char *)&v126 - v65;
      sub_2740590((__int64)(v103 + 1), v63 + 1, v63, v64, v67, v34);
      v65 = v103[1];
      v63 = *((unsigned int *)v103 + 4);
      v35 = v102;
      v34 = v97;
      v32 = v92;
      v66 = (char **)&v84[v65];
      v68 = *((_DWORD *)v103 + 4);
    }
  }
  v69 = 144 * v63;
  v70 = (_QWORD *)(v69 + v65);
  if ( v70 )
  {
    *v70 = v70 + 2;
    v70[1] = 0x800000000LL;
    if ( *((_DWORD *)v66 + 2) )
    {
      v91 = v32;
      v96 = v34;
      v101 = v35;
      sub_30ACEB0((__int64)v70, v66, v69, v64, v67, v34);
      v32 = v91;
      v34 = v96;
      v35 = v101;
    }
    v68 = *((_DWORD *)v103 + 4);
  }
  v71 = v68 + 1;
  *((_DWORD *)v103 + 4) = v71;
  v56 = v126;
  if ( v71 <= 0x1F4 )
  {
    if ( v126 != v128 )
    {
      v93 = v32;
      v98 = v34;
      v110 = v35;
      _libc_free((unsigned __int64)v126);
      v32 = v93;
      v34 = v98;
      v35 = v110;
    }
    goto LABEL_73;
  }
LABEL_44:
  if ( v56 != v128 )
    _libc_free((unsigned __int64)v56);
  v31 = (unsigned __int64 *)v129;
  v57 = 0;
  v58 = (unsigned __int64 *)&v129[144 * (unsigned int)v130];
LABEL_47:
  if ( v31 != v58 )
  {
    do
    {
      v58 -= 18;
      if ( (unsigned __int64 *)*v58 != v58 + 2 )
        _libc_free(*v58);
    }
    while ( v58 != v31 );
    v58 = (unsigned __int64 *)v129;
  }
  if ( v58 != (unsigned __int64 *)v131 )
    _libc_free((unsigned __int64)v58);
  return v57;
}
