// Function: sub_9B5220
// Address: 0x9b5220
//
__int64 __fastcall sub_9B5220(unsigned __int8 *a1, unsigned __int8 *a2, __int64 a3, unsigned int a4, __m128i *a5)
{
  unsigned int v5; // r12d
  unsigned __int8 *v6; // r15
  unsigned __int8 *v7; // r14
  int v9; // eax
  int v10; // edx
  unsigned int *v11; // rax
  int v12; // eax
  int v13; // edx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rax
  char v31; // al
  __int64 v32; // r11
  unsigned __int16 v33; // ax
  __int64 v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  char v37; // di
  _QWORD *v38; // r12
  char v39; // bl
  _BYTE *v40; // r8
  __int64 *v41; // rax
  _BYTE *v42; // rsi
  __int64 *v43; // rdx
  __int64 v44; // rdx
  char v45; // dl
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r11
  _BYTE *v53; // rdx
  char v54; // al
  __m128i v55; // xmm0
  _QWORD *v56; // r8
  __m128i v57; // xmm1
  __m128i v58; // xmm2
  unsigned __int64 v59; // xmm3_8
  unsigned __int64 v60; // rax
  int v61; // edx
  unsigned __int64 v62; // rax
  _BYTE *v63; // rax
  _BYTE *v64; // rax
  __int32 v65; // r12d
  __int32 v66; // r13d
  unsigned __int16 v67; // ax
  __int64 v68; // rdi
  __int64 v69; // r13
  __int64 v70; // rdx
  int v71; // r12d
  __int64 *v72; // rbx
  __int64 v73; // r10
  unsigned __int16 v74; // ax
  unsigned int v75; // eax
  _BYTE *v76; // [rsp+0h] [rbp-170h]
  __int64 v77; // [rsp+8h] [rbp-168h]
  __int64 v78; // [rsp+8h] [rbp-168h]
  __int64 v79; // [rsp+8h] [rbp-168h]
  __int64 v80; // [rsp+10h] [rbp-160h]
  __int64 v81; // [rsp+10h] [rbp-160h]
  _BYTE *v82; // [rsp+10h] [rbp-160h]
  _BYTE *v83; // [rsp+10h] [rbp-160h]
  unsigned __int8 v84; // [rsp+1Fh] [rbp-151h]
  unsigned __int8 v85; // [rsp+1Fh] [rbp-151h]
  unsigned __int64 v86; // [rsp+20h] [rbp-150h]
  unsigned __int64 v87; // [rsp+28h] [rbp-148h]
  unsigned __int64 v88; // [rsp+30h] [rbp-140h]
  unsigned int v89; // [rsp+38h] [rbp-138h]
  __int64 v90; // [rsp+40h] [rbp-130h]
  _BYTE *v91; // [rsp+40h] [rbp-130h]
  _BYTE *v92; // [rsp+40h] [rbp-130h]
  __int64 v93; // [rsp+40h] [rbp-130h]
  __int64 v94; // [rsp+40h] [rbp-130h]
  _QWORD *v95; // [rsp+48h] [rbp-128h]
  unsigned __int8 v96; // [rsp+48h] [rbp-128h]
  __int64 v98; // [rsp+50h] [rbp-120h]
  __int64 v100; // [rsp+58h] [rbp-118h]
  __int64 v101; // [rsp+60h] [rbp-110h] BYREF
  __int64 v102; // [rsp+68h] [rbp-108h] BYREF
  unsigned __int8 *v103; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int8 *v104; // [rsp+78h] [rbp-F8h]
  char v105; // [rsp+80h] [rbp-F0h]
  __m128i v106; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v107; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v108; // [rsp+B0h] [rbp-C0h]
  __int128 v109; // [rsp+C0h] [rbp-B0h]
  __int64 v110; // [rsp+D0h] [rbp-A0h]
  __int64 v111; // [rsp+E0h] [rbp-90h] BYREF
  __int64 *v112; // [rsp+E8h] [rbp-88h]
  __int64 v113; // [rsp+F0h] [rbp-80h] BYREF
  int v114; // [rsp+F8h] [rbp-78h]
  char v115; // [rsp+FCh] [rbp-74h]
  char v116; // [rsp+100h] [rbp-70h] BYREF

  if ( a2 == a1 )
    return 0;
  v6 = a1;
  v7 = a2;
  while ( 1 )
  {
    if ( *((_QWORD *)v6 + 1) != *((_QWORD *)v7 + 1) )
      return 0;
    v11 = (unsigned int *)sub_C94E20(qword_4F862D0);
    if ( v11 )
    {
      if ( *v11 <= a4 )
        return 0;
    }
    else if ( LODWORD(qword_4F862D0[2]) <= a4 )
    {
      return 0;
    }
    v12 = *v6;
    v13 = *v7;
    LOBYTE(v5) = (unsigned __int8)v12 > 0x1Cu || (_BYTE)v12 == 5;
    if ( !(_BYTE)v5 || (unsigned __int8)v13 <= 0x1Cu && (_BYTE)v13 != 5 )
      goto LABEL_15;
    v9 = (unsigned __int8)v12 <= 0x1Cu ? *((unsigned __int16 *)v6 + 1) : v12 - 29;
    v10 = (unsigned __int8)v13 <= 0x1Cu ? *((unsigned __int16 *)v7 + 1) : v13 - 29;
    if ( v9 != v10 )
      goto LABEL_15;
    sub_990F20((__int64)&v103, v6, (__int64)v7);
    if ( !v105 )
      break;
    v7 = v104;
    v6 = v103;
    ++a4;
    if ( v104 == v103 )
      return 0;
  }
  if ( *v6 != 84 || *((_QWORD *)v6 + 5) != *((_QWORD *)v7 + 5) )
    goto LABEL_15;
  v113 = 8;
  v111 = 0;
  v112 = (__int64 *)&v116;
  v114 = 0;
  v115 = 1;
  v34 = 32LL * *((unsigned int *)v6 + 18);
  v35 = v34 + 8LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
  v36 = (_QWORD *)(*((_QWORD *)v6 - 1) + v34);
  v95 = (_QWORD *)(v35 + *((_QWORD *)v6 - 1));
  if ( v95 == v36 )
    return v5;
  v85 = v5;
  v37 = v5;
  v38 = v36;
  v89 = a4;
  v39 = v105;
  while ( 1 )
  {
    v40 = (_BYTE *)*v38;
    if ( v37 )
    {
      v41 = v112;
      v42 = (_BYTE *)HIDWORD(v113);
      v43 = &v112[HIDWORD(v113)];
      if ( v112 != v43 )
      {
        while ( v40 != (_BYTE *)*v41 )
        {
          if ( v43 == ++v41 )
            goto LABEL_98;
        }
        goto LABEL_65;
      }
LABEL_98:
      if ( HIDWORD(v113) < (unsigned int)v113 )
        break;
    }
    v42 = (_BYTE *)*v38;
    v91 = (_BYTE *)*v38;
    sub_C8CC70(&v111, *v38);
    v37 = v115;
    v40 = v91;
    if ( v45 )
      goto LABEL_77;
LABEL_65:
    if ( v95 == ++v38 )
    {
      v5 = v85;
      if ( !v37 )
        _libc_free(v112, v42);
      return v5;
    }
  }
  ++HIDWORD(v113);
  *v43 = (__int64)v40;
  ++v111;
LABEL_77:
  v46 = *((_QWORD *)v6 - 1);
  v47 = 0x1FFFFFFFE0LL;
  if ( (*((_DWORD *)v6 + 1) & 0x7FFFFFF) != 0 )
  {
    v48 = 0;
    do
    {
      if ( v40 == *(_BYTE **)(v46 + 32LL * *((unsigned int *)v6 + 18) + 8 * v48) )
      {
        v47 = 32 * v48;
        goto LABEL_82;
      }
      ++v48;
    }
    while ( (*((_DWORD *)v6 + 1) & 0x7FFFFFF) != (_DWORD)v48 );
    v47 = 0x1FFFFFFFE0LL;
  }
LABEL_82:
  v49 = *(_QWORD *)(v46 + v47);
  v50 = 0x1FFFFFFFE0LL;
  v42 = (_BYTE *)*((_QWORD *)v7 - 1);
  if ( (*((_DWORD *)v7 + 1) & 0x7FFFFFF) != 0 )
  {
    v51 = 0;
    do
    {
      if ( v40 == *(_BYTE **)&v42[32 * *((unsigned int *)v7 + 18) + 8 * v51] )
      {
        v50 = 32 * v51;
        goto LABEL_87;
      }
      ++v51;
    }
    while ( (*((_DWORD *)v7 + 1) & 0x7FFFFFF) != (_DWORD)v51 );
    v50 = 0x1FFFFFFFE0LL;
  }
LABEL_87:
  v52 = *(_QWORD *)&v42[v50];
  v53 = (_BYTE *)(v49 + 24);
  if ( *(_BYTE *)v49 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v49 + 8) + 8LL) - 17 > 1 )
      goto LABEL_91;
    if ( *(_BYTE *)v49 > 0x15u )
      goto LABEL_91;
    v42 = 0;
    v78 = v52;
    v82 = v40;
    v93 = v49;
    v63 = (_BYTE *)sub_AD7630(v49, 0);
    v49 = v93;
    v40 = v82;
    v52 = v78;
    if ( !v63 || *v63 != 17 )
      goto LABEL_91;
    v53 = v63 + 24;
  }
  v42 = (_BYTE *)(v52 + 24);
  if ( *(_BYTE *)v52 != 17 )
  {
    v76 = v53;
    v42 = (_BYTE *)((unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v52 + 8) + 8LL) - 17);
    if ( (unsigned int)v42 > 1 )
      goto LABEL_91;
    if ( *(_BYTE *)v52 > 0x15u )
      goto LABEL_91;
    v79 = v49;
    v83 = v40;
    v94 = v52;
    v64 = (_BYTE *)sub_AD7630(v52, 0);
    v52 = v94;
    v40 = v83;
    v49 = v79;
    v42 = v64;
    if ( !v64 || *v64 != 17 )
      goto LABEL_91;
    v53 = v76;
    v42 = v64 + 24;
  }
  if ( *((_DWORD *)v53 + 2) <= 0x40u )
  {
    if ( *(_QWORD *)v53 == *(_QWORD *)v42 )
      goto LABEL_91;
LABEL_97:
    v37 = v115;
    goto LABEL_65;
  }
  v77 = v49;
  v81 = v52;
  v92 = v40;
  v54 = sub_C43C50(v53, v42);
  v40 = v92;
  v52 = v81;
  v49 = v77;
  if ( !v54 )
    goto LABEL_97;
LABEL_91:
  if ( !v39 )
  {
    v55 = _mm_loadu_si128(a5);
    v56 = v40 + 48;
    v57 = _mm_loadu_si128(a5 + 1);
    v58 = _mm_loadu_si128(a5 + 2);
    v59 = _mm_loadu_si128(a5 + 3).m128i_u64[0];
    v110 = a5[4].m128i_i64[0];
    v106 = v55;
    v109 = v59;
    v107 = v57;
    v108 = v58;
    v60 = *v56 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v60 == v56 )
    {
      v62 = 0;
    }
    else
    {
      if ( !v60 )
        BUG();
      v61 = *(unsigned __int8 *)(v60 - 24);
      v62 = v60 - 24;
      if ( (unsigned int)(v61 - 30) >= 0xB )
        v62 = 0;
    }
    v108.m128i_i64[1] = v62;
    v42 = (_BYTE *)v52;
    v39 = sub_9B5220(v49, v52, a3, v89 + 1, &v106);
    if ( v39 )
      goto LABEL_97;
  }
  a4 = v89;
  if ( !v115 )
    _libc_free(v112, v42);
LABEL_15:
  if ( (unsigned __int8)sub_9B4FC0(v6, (__int64)v7, a3, a4, a5)
    || (unsigned __int8)sub_9B4FC0(v7, (__int64)v6, a3, a4, a5) )
  {
    return 1;
  }
  v15 = *v7;
  if ( (unsigned __int8)v15 <= 0x1Cu )
  {
    if ( (_BYTE)v15 == 5 && ((*((_WORD *)v7 + 1) & 0xFFF7) == 0x11 || (*((_WORD *)v7 + 1) & 0xFFFD) == 0xD) )
      goto LABEL_28;
  }
  else if ( (unsigned __int8)v15 <= 0x36u )
  {
    v16 = 0x40540000000000LL;
    if ( _bittest64(&v16, v15) )
    {
LABEL_28:
      if ( (unsigned __int8)sub_9B5030((__int64)v6, v7, a3, a4, a5) )
        return 1;
    }
  }
  v17 = *v6;
  if ( (unsigned __int8)v17 > 0x1Cu )
  {
    if ( (unsigned __int8)v17 > 0x36u )
      goto LABEL_31;
    v44 = 0x40540000000000LL;
    if ( !_bittest64(&v44, v17) )
      goto LABEL_31;
  }
  else if ( (_BYTE)v17 != 5 || (*((_WORD *)v6 + 1) & 0xFFFD) != 0xD && (*((_WORD *)v6 + 1) & 0xFFF7) != 0x11 )
  {
    goto LABEL_31;
  }
  if ( (unsigned __int8)sub_9B5030((__int64)v7, v6, a3, a4, a5) )
    return 1;
LABEL_31:
  if ( (unsigned __int8)sub_987880(v7) && (unsigned __int8)sub_9B5130((__int64)v6, v7, a3, a4, a5)
    || (unsigned __int8)sub_987880(v6) && (unsigned __int8)sub_9B5130((__int64)v7, v6, a3, a4, a5) )
  {
    return 1;
  }
  v18 = *((_QWORD *)v6 + 1);
  v19 = *(unsigned __int8 *)(v18 + 8);
  if ( (unsigned int)(v19 - 17) <= 1 )
    LOBYTE(v19) = *(_BYTE *)(**(_QWORD **)(v18 + 16) + 8LL);
  if ( (_BYTE)v19 == 12 )
  {
    sub_9B0110((__int64)&v106, (__int64)v6, a3, a4, a5);
    v65 = v106.m128i_i32[2];
    if ( v106.m128i_i32[2] <= 0x40u )
    {
      if ( v106.m128i_i64[0] )
        goto LABEL_126;
    }
    else if ( v65 != (unsigned int)sub_C444A0(&v106) )
    {
      goto LABEL_126;
    }
    v66 = v107.m128i_i32[2];
    if ( v107.m128i_i32[2] <= 0x40u )
    {
      if ( !v107.m128i_i64[0] )
        goto LABEL_131;
    }
    else if ( v66 == (unsigned int)sub_C444A0(&v107) )
    {
      goto LABEL_131;
    }
LABEL_126:
    sub_9B0110((__int64)&v111, (__int64)v7, a3, a4, a5);
    if ( v106.m128i_i32[2] <= 0x40u )
    {
      if ( (v113 & v106.m128i_i64[0]) != 0 )
        goto LABEL_142;
    }
    else if ( (unsigned __int8)sub_C446A0(&v106, &v113) )
    {
      goto LABEL_142;
    }
    if ( (unsigned int)v112 <= 0x40 )
    {
      if ( (v107.m128i_i64[0] & v111) == 0 )
        goto LABEL_130;
    }
    else if ( !(unsigned __int8)sub_C446A0(&v111, &v107) )
    {
LABEL_130:
      sub_969240(&v113);
      sub_969240(&v111);
LABEL_131:
      sub_969240(v107.m128i_i64);
      sub_969240(v106.m128i_i64);
      goto LABEL_36;
    }
LABEL_142:
    sub_969240(&v113);
    sub_969240(&v111);
    v5 = 1;
    sub_969240(v107.m128i_i64);
    sub_969240(v106.m128i_i64);
    return v5;
  }
LABEL_36:
  if ( (unsigned __int8)sub_9B5F80(v6, v7, a3, a4, a5) )
    return 1;
  v5 = sub_9B5F80(v7, v6, a3, a4, a5);
  if ( (_BYTE)v5
    || *(_BYTE *)(*((_QWORD *)v6 + 1) + 8LL) == 14
    && *(_BYTE *)(*((_QWORD *)v7 + 1) + 8LL) == 14
    && ((unsigned __int8)sub_986C90((char *)v6, (int)v7, a5)
     || *(_BYTE *)(*((_QWORD *)v7 + 1) + 8LL) == 14
     && *(_BYTE *)(*((_QWORD *)v6 + 1) + 8LL) == 14
     && (unsigned __int8)sub_986C90((char *)v7, (int)v6, a5)) )
  {
    return 1;
  }
  v106.m128i_i64[0] = a5->m128i_i64[0];
  v106.m128i_i64[1] = (__int64)&v101;
  if ( (unsigned __int8)sub_994000((__int64)&v106, v6)
    && (v111 = a5->m128i_i64[0], v112 = &v102, (unsigned __int8)sub_994000((__int64)&v111, v7)) )
  {
    return (unsigned int)sub_9B5220(v101, v102, a3, a4 + 1, a5);
  }
  else if ( a5[2].m128i_i64[1] )
  {
    v20 = a5[3].m128i_i64[0];
    if ( v20 )
    {
      if ( a5[1].m128i_i64[1] )
      {
        v21 = sub_988120(v20, (__int64)v6);
        v90 = v21 + 8 * v22;
        if ( v21 != v90 )
        {
          v23 = v21;
          v84 = v5;
          while ( 1 )
          {
            v27 = a5[1].m128i_i64[1];
            v28 = *(_QWORD *)(*(_QWORD *)v23 + 40LL);
            v80 = *(_QWORD *)v23;
            v29 = *(_QWORD *)(*(_QWORD *)v23 - 96LL);
            v106.m128i_i64[1] = *(_QWORD *)(*(_QWORD *)v23 - 32LL);
            v30 = a5[2].m128i_i64[1];
            v106.m128i_i64[0] = v28;
            v31 = sub_B19C20(v27, &v106, *(_QWORD *)(v30 + 40));
            v32 = v80;
            if ( v31 )
            {
              v86 = v86 & 0xFFFFFF0000000000LL | 0x21;
              v33 = sub_9A13D0(v29, v86, (__int64)v6, v7, a5->m128i_i64[0], 1u, a4);
              v32 = v80;
              if ( HIBYTE(v33) )
              {
                if ( (_BYTE)v33 )
                  return 1;
              }
            }
            v24 = *(_QWORD *)(v32 + 40);
            v25 = a5[1].m128i_i64[1];
            v112 = *(__int64 **)(v32 - 64);
            v26 = a5[2].m128i_i64[1];
            v111 = v24;
            if ( (unsigned __int8)sub_B19C20(v25, &v111, *(_QWORD *)(v26 + 40)) )
            {
              v87 = v87 & 0xFFFFFF0000000000LL | 0x21;
              v67 = sub_9A13D0(v29, v87, (__int64)v6, v7, a5->m128i_i64[0], 0, a4);
              if ( HIBYTE(v67) )
              {
                if ( (_BYTE)v67 )
                  return 1;
              }
            }
            v23 += 8;
            if ( v90 == v23 )
            {
              v5 = v84;
              break;
            }
          }
        }
      }
    }
    v68 = a5[2].m128i_i64[0];
    if ( v68 )
    {
      v69 = sub_988050(v68, (__int64)v6);
      v70 *= 32;
      v98 = v69 + v70;
      if ( v69 != v69 + v70 )
      {
        v96 = v5;
        v71 = a4;
        v72 = (__int64 *)a5;
        do
        {
          v73 = *(_QWORD *)(v69 + 16);
          if ( v73 )
          {
            v100 = *(_QWORD *)(v69 + 16);
            v88 = v88 & 0xFFFFFF0000000000LL | 0x21;
            v74 = sub_9A13D0(
                    *(_QWORD *)(v73 - 32LL * (*(_DWORD *)(v73 + 4) & 0x7FFFFFF)),
                    v88,
                    (__int64)v6,
                    v7,
                    *v72,
                    1u,
                    v71);
            if ( HIBYTE(v74) )
            {
              if ( (_BYTE)v74 )
              {
                v75 = sub_98CF40(v100, v72[5], v72[3], 0);
                if ( (_BYTE)v75 )
                  return v75;
              }
            }
          }
          v69 += 32;
        }
        while ( v69 != v98 );
        return v96;
      }
    }
  }
  return v5;
}
