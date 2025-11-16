// Function: sub_21C09D0
// Address: 0x21c09d0
//
__int64 __fastcall sub_21C09D0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  const __m128i *v6; // r12
  __int64 v7; // rax
  _QWORD *v8; // r9
  __int64 v9; // rax
  _QWORD *v10; // r14
  unsigned int v11; // r8d
  unsigned __int16 v12; // ax
  __m128i v13; // xmm1
  __int64 v14; // rdx
  __m128i v15; // xmm2
  int v16; // eax
  __int64 *v17; // rdx
  const __m128i *v18; // r8
  __int64 v19; // r12
  __int64 v20; // r11
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdi
  unsigned int v28; // edx
  int v29; // r9d
  __int64 v30; // r14
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int16 v37; // ax
  __int128 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rax
  __int16 v41; // r14
  __int64 v42; // r8
  __int64 *v43; // rax
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r9
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned __int8 v54; // r14
  __int64 v55; // rax
  __int16 v56; // dx
  char *v57; // rdx
  __int64 v58; // rsi
  __int64 *v59; // rdi
  const void **v60; // r8
  unsigned __int8 v61; // dl
  __int64 v62; // r8
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 *v66; // rcx
  __int128 v67; // rax
  __int64 v68; // r9
  char v69; // r14
  __int64 *v70; // r9
  __int64 v71; // rdx
  __int64 v72; // rcx
  int v73; // r9d
  int v74; // r8d
  char v75; // r14
  __int64 *v76; // r9
  _QWORD *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // [rsp-10h] [rbp-2B0h]
  int v80; // [rsp-8h] [rbp-2A8h]
  __int64 v81; // [rsp+8h] [rbp-298h]
  unsigned __int8 v82; // [rsp+8h] [rbp-298h]
  unsigned __int8 v83; // [rsp+8h] [rbp-298h]
  unsigned __int8 v84; // [rsp+8h] [rbp-298h]
  unsigned __int8 v85; // [rsp+8h] [rbp-298h]
  const __m128i *v86; // [rsp+10h] [rbp-290h]
  char v87; // [rsp+10h] [rbp-290h]
  char v88; // [rsp+10h] [rbp-290h]
  _QWORD *v89; // [rsp+18h] [rbp-288h]
  __int64 v90; // [rsp+18h] [rbp-288h]
  __m128i v91; // [rsp+20h] [rbp-280h] BYREF
  __m128i v92; // [rsp+30h] [rbp-270h] BYREF
  __int64 *v93; // [rsp+40h] [rbp-260h]
  __int64 *v94; // [rsp+48h] [rbp-258h]
  __int64 v95; // [rsp+50h] [rbp-250h]
  __int64 v96; // [rsp+58h] [rbp-248h]
  __int64 v97; // [rsp+60h] [rbp-240h]
  __int64 v98; // [rsp+68h] [rbp-238h]
  __int64 v99; // [rsp+70h] [rbp-230h]
  __int64 v100; // [rsp+78h] [rbp-228h]
  __int64 v101; // [rsp+88h] [rbp-218h] BYREF
  __int64 v102; // [rsp+90h] [rbp-210h] BYREF
  __int64 v103; // [rsp+98h] [rbp-208h] BYREF
  __int64 v104; // [rsp+A0h] [rbp-200h] BYREF
  int v105; // [rsp+A8h] [rbp-1F8h]
  __int64 v106; // [rsp+B0h] [rbp-1F0h] BYREF
  int v107; // [rsp+B8h] [rbp-1E8h]
  __int64 *v108; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-1D8h]
  _BYTE v110[128]; // [rsp+D0h] [rbp-1D0h] BYREF
  __m128i v111; // [rsp+150h] [rbp-150h] BYREF
  _BYTE v112[128]; // [rsp+160h] [rbp-140h] BYREF
  _BYTE *v113; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v114; // [rsp+1E8h] [rbp-B8h]
  _BYTE v115[176]; // [rsp+1F0h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 72);
  v104 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v104, v5, 2);
  v6 = *(const __m128i **)(a2 + 32);
  v105 = *(_DWORD *)(a2 + 64);
  v7 = *(_QWORD *)(v6[2].m128i_i64[1] + 88);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = *(_QWORD *)(v6[5].m128i_i64[0] + 88);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = 0;
  v12 = *(_WORD *)(a2 + 24) - 670;
  if ( v12 <= 4u )
  {
    v13 = _mm_loadu_si128(v6);
    v14 = (unsigned int)(*(_DWORD *)(a2 + 56) - 1);
    v92 = v13;
    v15 = _mm_loadu_si128((const __m128i *)((char *)v6 + 40 * v14));
    v108 = (__int64 *)v110;
    v16 = dword_433D7D0[v12];
    v109 = 0x800000000LL;
    v91 = v15;
    LODWORD(v94) = v16;
    if ( v16 )
    {
      v17 = (__int64 *)v110;
      v18 = (const __m128i *)((char *)v6 + 120);
      v19 = 160;
      v20 = 40LL * (unsigned int)((_DWORD)v94 - 1) + 160;
      v21 = 0;
      while ( 1 )
      {
        a3 = _mm_loadu_si128(v18);
        *(__m128i *)&v17[2 * v21] = a3;
        v21 = (unsigned int)(v109 + 1);
        LODWORD(v109) = v109 + 1;
        if ( v19 == v20 )
          break;
        v18 = (const __m128i *)(v19 + *(_QWORD *)(a2 + 32));
        if ( HIDWORD(v109) <= (unsigned int)v21 )
        {
          v81 = v20;
          v86 = (const __m128i *)(v19 + *(_QWORD *)(a2 + 32));
          v89 = v8;
          v93 = (__int64 *)&v108;
          sub_16CD150((__int64)&v108, v110, 0, 16, (int)v18, (int)v8);
          v21 = (unsigned int)v109;
          v20 = v81;
          v18 = v86;
          v8 = v89;
        }
        v17 = v108;
        v19 += 40;
      }
    }
    v23 = sub_1D38BB0(
            *(_QWORD *)(a1 + 272),
            (unsigned int)v8,
            (__int64)&v104,
            5,
            0,
            1,
            a3,
            *(double *)v13.m128i_i64,
            v15,
            0);
    v24 = v22;
    v25 = (unsigned int)v109;
    if ( (unsigned int)v109 >= HIDWORD(v109) )
    {
      v90 = v23;
      v93 = (__int64 *)v22;
      sub_16CD150((__int64)&v108, v110, 0, 16, v23, v22);
      v25 = (unsigned int)v109;
      v23 = v90;
      v24 = (__int64)v93;
    }
    v26 = &v108[2 * v25];
    *v26 = v23;
    v26[1] = v24;
    v27 = *(_QWORD *)(a1 + 272);
    LODWORD(v109) = v109 + 1;
    v30 = sub_1D38BB0(v27, (unsigned int)v10, (__int64)&v104, 5, 0, 1, a3, *(double *)v13.m128i_i64, v15, 0);
    v31 = v28;
    v32 = (unsigned int)v109;
    if ( (unsigned int)v109 >= HIDWORD(v109) )
    {
      v93 = (__int64 *)v28;
      sub_16CD150((__int64)&v108, v110, 0, 16, v28, v29);
      v32 = (unsigned int)v109;
      v31 = (__int64)v93;
    }
    v33 = &v108[2 * v32];
    *v33 = v30;
    v33[1] = v31;
    v34 = (unsigned int)(v109 + 1);
    LODWORD(v109) = v34;
    if ( HIDWORD(v109) <= (unsigned int)v34 )
    {
      sub_16CD150((__int64)&v108, v110, 0, 16, v31, v29);
      v34 = (unsigned int)v109;
    }
    *(__m128i *)&v108[2 * v34] = _mm_load_si128(&v92);
    v35 = (unsigned int)(v109 + 1);
    LODWORD(v109) = v35;
    if ( HIDWORD(v109) <= (unsigned int)v35 )
    {
      sub_16CD150((__int64)&v108, v110, 0, 16, v31, v29);
      v35 = (unsigned int)v109;
    }
    *(__m128i *)&v108[2 * v35] = _mm_load_si128(&v91);
    v36 = (unsigned int)(v109 + 1);
    v37 = *(_WORD *)(a2 + 24);
    LODWORD(v109) = v109 + 1;
    if ( v37 == 673 )
    {
      *(_QWORD *)&v67 = sub_1D38BB0(
                          *(_QWORD *)(a1 + 272),
                          0,
                          (__int64)&v104,
                          5,
                          0,
                          1,
                          a3,
                          *(double *)v13.m128i_i64,
                          v15,
                          0);
      v40 = sub_1D2CCE0(*(_QWORD **)(a1 + 272), 254, (__int64)&v104, 5, 0, v68, *(_OWORD *)v108, v67);
      goto LABEL_25;
    }
    if ( v37 == 674 )
    {
      *(_QWORD *)&v38 = sub_1D38BB0(
                          *(_QWORD *)(a1 + 272),
                          0,
                          (__int64)&v104,
                          5,
                          0,
                          1,
                          a3,
                          *(double *)v13.m128i_i64,
                          v15,
                          0);
      v40 = sub_1D2CCE0(*(_QWORD **)(a1 + 272), 302, (__int64)&v104, 5, 0, v39, *(_OWORD *)v108, v38);
LABEL_25:
      v41 = 4108;
      v42 = v40;
      v43 = v108;
      *v108 = v42;
      *((_DWORD *)v43 + 2) = 0;
      goto LABEL_26;
    }
    if ( (_DWORD)v94 == 2 )
    {
      v69 = *(_BYTE *)(a2 + 88);
      v91.m128i_i64[0] = (__int64)v112;
      v111.m128i_i64[0] = (__int64)v112;
      v111.m128i_i64[1] = 0x800000000LL;
      v106 = v104;
      v94 = &v106;
      if ( v104 )
      {
        sub_1623A60((__int64)&v106, v104, 2);
        v36 = (unsigned int)v109;
      }
      v70 = *(__int64 **)(a1 + 272);
      v114 = 0x800000000LL;
      v107 = v105;
      v87 = v69;
      v92.m128i_i64[0] = (__int64)&v113;
      v113 = v115;
      if ( (_DWORD)v36 )
      {
        v93 = v70;
        sub_21BD6C0(v92.m128i_i64[0], (__int64)&v108, v36, 0x800000000LL, v31, (int)v70);
        v70 = v93;
      }
      v41 = 0;
      v93 = &v103;
      sub_21C01C0((__int64)&v103, v92.m128i_i64[0], 2, v87, &v111, v70, a3, *(double *)v13.m128i_i64, v15, (__int64)v94);
      v73 = v80;
      v74 = BYTE4(v103);
      if ( BYTE4(v103) )
        v41 = v103;
      if ( v113 != v115 )
      {
        v82 = BYTE4(v103);
        _libc_free((unsigned __int64)v113);
        v74 = v82;
      }
      if ( v106 )
      {
        v83 = v74;
        sub_161E7C0((__int64)v94, v106);
        v74 = v83;
      }
      if ( !(_BYTE)v74 )
      {
        v106 = 0x100001018LL;
        v103 = 0x100001016LL;
        v102 = 0x100001015LL;
        v101 = 0x10000101BLL;
        sub_21BD570(
          v92.m128i_i64[0],
          v87,
          4124,
          4121,
          4122,
          (__int64)&v101,
          (__int64)&v102,
          (__int64)v93,
          4119,
          (__int64)v94);
        goto LABEL_58;
      }
    }
    else
    {
      if ( (_DWORD)v94 != 4 )
      {
        v11 = 0;
        if ( (_DWORD)v94 != 1 )
          goto LABEL_27;
        v54 = *(_BYTE *)(a2 + 88);
        if ( v54 == 8
          || v54 == 86
          || (v55 = *v108, v56 = *(_WORD *)(*v108 + 24), (unsigned __int16)(v56 - 10) > 1u)
          && (unsigned __int16)(v56 - 32) > 1u )
        {
          v111.m128i_i64[0] = 0x10000100ALL;
          v106 = 0x100001008LL;
          v103 = 0x100001007LL;
          v102 = 0x10000100DLL;
          sub_21BD570(
            (__int64)&v113,
            v54,
            4110,
            4107,
            4108,
            (__int64)&v102,
            (__int64)&v103,
            (__int64)&v106,
            4105,
            (__int64)&v111);
        }
        else
        {
          v57 = *(char **)(v55 + 40);
          v58 = *(_QWORD *)(v55 + 88);
          v59 = *(__int64 **)(a1 + 272);
          v60 = (const void **)*((_QWORD *)v57 + 1);
          v61 = *v57;
          if ( (unsigned __int8)(v54 - 9) <= 1u )
          {
            v77 = sub_1D360F0(
                    v59,
                    v58,
                    (__int64)&v104,
                    v61,
                    v60,
                    1,
                    *(double *)a3.m128i_i64,
                    *(double *)v13.m128i_i64,
                    v15);
            v100 = v78;
            v64 = (__int64)v77;
            v65 = (unsigned int)v100;
            v99 = v64;
          }
          else
          {
            v62 = sub_1D37E40((__int64)v59, v58, (__int64)&v104, v61, v60, 1, a3, *(double *)v13.m128i_i64, v15, 0);
            v65 = v63;
            v97 = v62;
            v64 = v62;
            v98 = v65;
            v65 = (unsigned int)v65;
          }
          v66 = v108;
          v95 = v64;
          v96 = v65;
          *v108 = v64;
          *((_DWORD *)v66 + 2) = v96;
          v111.m128i_i64[0] = 0x100001010LL;
          v102 = 0x100001013LL;
          BYTE4(v106) = 0;
          BYTE4(v103) = 0;
          sub_21BD570(
            (__int64)&v113,
            v54,
            4116,
            4113,
            4114,
            (__int64)&v102,
            (__int64)&v103,
            (__int64)&v106,
            4111,
            (__int64)&v111);
        }
        v11 = BYTE4(v113);
        if ( !BYTE4(v113) )
        {
LABEL_27:
          if ( v108 != (__int64 *)v110 )
          {
            LOBYTE(v94) = v11;
            _libc_free((unsigned __int64)v108);
            v11 = (unsigned __int8)v94;
          }
          goto LABEL_29;
        }
        v41 = (__int16)v113;
LABEL_26:
        v44 = sub_1D252B0(*(_QWORD *)(a1 + 272), 1, 0, 111, 0);
        v47 = sub_1D23DE0(*(_QWORD **)(a1 + 272), v41, (__int64)&v104, v44, v45, v46, v108, (unsigned int)v109);
        v48 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
        *v48 = *(_QWORD *)(a2 + 104);
        *(_QWORD *)(v47 + 88) = v48;
        *(_QWORD *)(v47 + 96) = v48 + 1;
        sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v47);
        sub_1D49010(v47);
        sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v49, v50, v51, v52);
        v11 = 1;
        goto LABEL_27;
      }
      v75 = *(_BYTE *)(a2 + 88);
      v91.m128i_i64[0] = (__int64)v112;
      v111.m128i_i64[0] = (__int64)v112;
      v111.m128i_i64[1] = 0x800000000LL;
      v106 = v104;
      v94 = &v106;
      if ( v104 )
      {
        sub_1623A60((__int64)&v106, v104, 2);
        v36 = (unsigned int)v109;
      }
      v76 = *(__int64 **)(a1 + 272);
      v114 = 0x800000000LL;
      v107 = v105;
      v88 = v75;
      v92.m128i_i64[0] = (__int64)&v113;
      v113 = v115;
      if ( (_DWORD)v36 )
      {
        v93 = v76;
        sub_21BD6C0(v92.m128i_i64[0], (__int64)&v108, v36, 0x800000000LL, v31, (int)v76);
        v76 = v93;
      }
      v41 = 0;
      v93 = &v103;
      sub_21C01C0((__int64)&v103, v92.m128i_i64[0], 4, v88, &v111, v76, a3, *(double *)v13.m128i_i64, v15, (__int64)v94);
      v74 = BYTE4(v103);
      v72 = v79;
      if ( BYTE4(v103) )
        v41 = v103;
      if ( v113 != v115 )
      {
        v84 = BYTE4(v103);
        _libc_free((unsigned __int64)v113);
        v74 = v84;
      }
      if ( v106 )
      {
        v85 = v74;
        sub_161E7C0((__int64)v94, v106);
        v74 = v85;
      }
      if ( !(_BYTE)v74 )
      {
        v103 = 0x100001036LL;
        v102 = 0x100001035LL;
        BYTE4(v106) = 0;
        BYTE4(v101) = 0;
        sub_21BD570(
          v92.m128i_i64[0],
          v88,
          4154,
          4152,
          4153,
          (__int64)&v101,
          (__int64)&v102,
          (__int64)v93,
          4151,
          (__int64)v94);
LABEL_58:
        v11 = BYTE4(v113);
        if ( BYTE4(v113) )
          v41 = (__int16)v113;
LABEL_60:
        if ( v111.m128i_i64[0] != v91.m128i_i64[0] )
        {
          LOBYTE(v94) = v11;
          _libc_free(v111.m128i_u64[0]);
          v11 = (unsigned __int8)v94;
        }
        if ( !(_BYTE)v11 )
          goto LABEL_27;
        goto LABEL_26;
      }
    }
    LOBYTE(v94) = v74;
    sub_21BD6C0((__int64)&v108, (__int64)&v111, v71, v72, v74, v73);
    v11 = (unsigned __int8)v94;
    goto LABEL_60;
  }
LABEL_29:
  if ( v104 )
  {
    LOBYTE(v94) = v11;
    sub_161E7C0((__int64)&v104, v104);
    return (unsigned __int8)v94;
  }
  return v11;
}
