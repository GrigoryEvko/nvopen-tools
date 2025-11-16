// Function: sub_A13C30
// Address: 0xa13c30
//
_QWORD *__fastcall sub_A13C30(__int64 *a1, __m128i *a2, char a3)
{
  __m128i *v3; // r14
  _QWORD *v4; // r12
  char v5; // bl
  __int64 v6; // rsi
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rdi
  __int64 *v18; // rbx
  unsigned __int64 v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rsi
  char v22; // dl
  __int64 m128i_i64; // rsi
  __int64 v24; // rdx
  __int64 v25; // rdi
  int v26; // edx
  unsigned __int64 v27; // rcx
  unsigned __int64 i; // rdx
  __int32 v29; // r13d
  _QWORD *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 *v33; // r12
  _BYTE *v34; // rax
  unsigned __int8 v35; // dl
  _BYTE **v36; // rbx
  __int64 v37; // rsi
  _BYTE **v38; // r13
  __int64 v39; // r15
  _BYTE *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r15
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rbx
  volatile signed __int32 *v49; // rdi
  signed __int32 v50; // eax
  void (*v51)(void); // rax
  signed __int32 v52; // eax
  void (*v53)(void); // rax
  __int64 *v54; // r13
  __int64 *v55; // rbx
  _BYTE *v56; // rdx
  unsigned __int8 v57; // cl
  __int64 v58; // rax
  __int64 v59; // rsi
  _BYTE **v60; // r14
  __int64 v61; // r15
  _BYTE **v62; // r12
  _BYTE *v63; // rdi
  __int64 v64; // rax
  _QWORD *v65; // rbx
  __int64 v66; // rdi
  __int64 v67; // rcx
  unsigned __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // r13
  __int64 v71; // rbx
  __m128i *v72; // r15
  _QWORD *v73; // r14
  volatile signed __int32 *v74; // r12
  signed __int32 v75; // edx
  void (*v76)(); // rdx
  signed __int32 v77; // edx
  __int64 (__fastcall *v78)(__int64); // rdx
  __int64 *v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int32 v84; // eax
  __int64 v85; // rbx
  unsigned __int64 v86; // r12
  __int64 v87; // rsi
  __int64 v88; // [rsp-8h] [rbp-348h]
  unsigned int v89; // [rsp+18h] [rbp-328h]
  _QWORD *v90; // [rsp+18h] [rbp-328h]
  __int64 v91; // [rsp+18h] [rbp-328h]
  __int32 v92; // [rsp+18h] [rbp-328h]
  int v93; // [rsp+20h] [rbp-320h]
  __int64 v94; // [rsp+20h] [rbp-320h]
  __int64 *v95; // [rsp+20h] [rbp-320h]
  __int64 v96; // [rsp+20h] [rbp-320h]
  char v97; // [rsp+20h] [rbp-320h]
  _QWORD *v98; // [rsp+20h] [rbp-320h]
  unsigned int v99; // [rsp+38h] [rbp-308h]
  __m128i *v100; // [rsp+38h] [rbp-308h]
  __int64 v101; // [rsp+40h] [rbp-300h]
  unsigned int v103; // [rsp+54h] [rbp-2ECh] BYREF
  __m128i v104; // [rsp+58h] [rbp-2E8h] BYREF
  unsigned __int64 v105; // [rsp+68h] [rbp-2D8h]
  __int64 v106; // [rsp+70h] [rbp-2D0h]
  __int64 v107; // [rsp+78h] [rbp-2C8h]
  __int64 v108; // [rsp+80h] [rbp-2C0h] BYREF
  __int64 v109; // [rsp+88h] [rbp-2B8h]
  __int64 v110; // [rsp+90h] [rbp-2B0h]
  unsigned __int64 v111; // [rsp+98h] [rbp-2A8h]
  char v112; // [rsp+A0h] [rbp-2A0h]
  char v113; // [rsp+A1h] [rbp-29Fh]
  __int64 v114; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v115; // [rsp+B8h] [rbp-288h]
  __int64 v116; // [rsp+C0h] [rbp-280h]
  __int64 v117; // [rsp+C8h] [rbp-278h]
  __int64 v118; // [rsp+D0h] [rbp-270h]
  unsigned __int64 v119; // [rsp+D8h] [rbp-268h]
  __int64 v120; // [rsp+E0h] [rbp-260h]
  unsigned __int64 v121; // [rsp+E8h] [rbp-258h]
  __int64 v122; // [rsp+F0h] [rbp-250h]
  __int64 v123; // [rsp+F8h] [rbp-248h]
  unsigned __int64 v124; // [rsp+100h] [rbp-240h] BYREF
  __int64 v125; // [rsp+108h] [rbp-238h]
  _BYTE v126[560]; // [rsp+110h] [rbp-230h] BYREF

  v3 = a2;
  v4 = a1;
  v5 = a3;
  if ( !a3 && (unsigned __int32)a2[2].m128i_i32[0] >> 1 )
  {
    v126[17] = 1;
    v124 = (unsigned __int64)"Invalid metadata: fwd refs into function blocks";
    v126[16] = 3;
    sub_A01DB0(a1, (__int64)&v124);
    return v4;
  }
  v6 = a2[15].m128i_i64[0];
  v101 = *(_QWORD *)(v6 + 16);
  v99 = *(_DWORD *)(v6 + 32);
  sub_A4DCE0(&v124, v6, 15, 0);
  if ( (v124 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v124 & 0xFFFFFFFFFFFFFFFELL | 1;
    return v4;
  }
  v114 = 0;
  v124 = (unsigned __int64)v126;
  v125 = 0x4000000000LL;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  sub_A04120(&v114, 0);
  LODWORD(v9) = v3->m128i_i32[2];
  if ( !v5 || !v3[71].m128i_i8[0] )
  {
    v10 = (__int64)&v104.m128i_i64[1];
LABEL_8:
    v103 = v9;
    while ( 1 )
    {
      sub_9CEFB0((__int64)&v108, v3[15].m128i_i64[0], 0, v10);
      if ( (v109 & 1) != 0 )
      {
        LOBYTE(v109) = v109 & 0xFD;
        v14 = v108;
        v108 = 0;
        v104.m128i_i64[1] = v14 | 1;
      }
      else
      {
        v104.m128i_i64[1] = 1;
        v89 = HIDWORD(v108);
        v93 = v108;
      }
      v15 = v104.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v104.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_12;
      if ( v93 == 1 )
      {
        sub_A10370(v3, (__int64)&v114, v11, v12, v13);
        if ( v3[48].m128i_i64[0] != v3[48].m128i_i64[1] )
        {
          v100 = v3;
          v54 = (__int64 *)v3[48].m128i_i64[1];
          v97 = v5;
          v55 = (__int64 *)v3[48].m128i_i64[0];
          do
          {
            v56 = (_BYTE *)v55[1];
            if ( v56 && *v56 == 5 )
            {
              v57 = *(v56 - 16);
              if ( (v57 & 2) != 0 )
              {
                v58 = *((_QWORD *)v56 - 4);
                v59 = *((unsigned int *)v56 - 6);
              }
              else
              {
                v59 = (*((_WORD *)v56 - 8) >> 6) & 0xF;
                v58 = (__int64)&v56[-8 * ((v57 >> 2) & 0xF) - 16];
              }
              v60 = (_BYTE **)(v58 + 8 * v59);
              if ( (_BYTE **)v58 != v60 )
              {
                v61 = *v55;
                v62 = (_BYTE **)v58;
                do
                {
                  v63 = *v62;
                  if ( *v62 && *v63 == 18 )
                    sub_BA6610(v63, 5, v61);
                  ++v62;
                }
                while ( v60 != v62 );
              }
            }
            v55 += 2;
          }
          while ( v54 != v55 );
          v3 = v100;
          v4 = a1;
          v5 = v97;
          v64 = v100[48].m128i_i64[0];
          if ( v100[48].m128i_i64[1] != v64 )
            v100[48].m128i_i64[1] = v64;
        }
        sub_A027C0((__int64)v3);
        if ( v5 )
          sub_A12E50((__int64)v3);
        *v4 = 1;
        goto LABEL_13;
      }
      if ( (v93 & 0xFFFFFFFD) == 0 )
      {
        v113 = 1;
        v108 = (__int64)"Malformed block";
        v112 = 3;
        sub_A01DB0(a1, (__int64)&v108);
        goto LABEL_13;
      }
      v21 = v3[15].m128i_i64[0];
      LODWORD(v125) = 0;
      v104.m128i_i64[1] = 0;
      v105 = 0;
      sub_A4B600(&v108, v21, v89, &v124, &v104.m128i_u64[1]);
      v22 = v109 & 1;
      LOBYTE(v109) = (2 * (v109 & 1)) | v109 & 0xFD;
      if ( v22 )
        goto LABEL_105;
      m128i_i64 = (__int64)v3;
      sub_A09F80(
        &v104,
        v3,
        (__int64 **)&v124,
        (_BYTE *)(unsigned int)v108,
        (__int64)&v114,
        &v103,
        v104.m128i_i64[1],
        v105);
      v24 = v88;
      if ( (v104.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = v104.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        if ( (v109 & 2) != 0 )
LABEL_103:
          sub_9CE230(&v108);
        if ( (v109 & 1) == 0 )
          goto LABEL_13;
        v25 = v108;
        if ( !v108 )
          goto LABEL_13;
        goto LABEL_28;
      }
      if ( (v109 & 2) != 0 )
        goto LABEL_103;
      if ( (v109 & 1) != 0 && v108 )
        (*(void (__fastcall **)(__int64, __m128i *, __int64))(*(_QWORD *)v108 + 8LL))(v108, v3, v88);
    }
  }
  v10 = (__int64)&v104.m128i_i64[1];
  if ( (_DWORD)v9 || byte_4F80548 )
    goto LABEL_8;
  sub_A07720((__int64)&v104.m128i_i64[1], v3, v8, (__int64)&v104.m128i_i64[1]);
  v26 = v105 & 1;
  v10 = (unsigned int)(2 * v26);
  LOBYTE(v105) = (2 * v26) | v105 & 0xFD;
  if ( (_BYTE)v26 )
  {
    *a1 = v104.m128i_i64[1] | 1;
    goto LABEL_13;
  }
  v9 = v3->m128i_u32[2];
  if ( !v104.m128i_i8[8] )
    goto LABEL_8;
  v27 = (unsigned __int64)&v108;
  i = (unsigned int)((v3[46].m128i_i64[1] - v3[46].m128i_i64[0]) >> 3)
    + (unsigned int)((v3[45].m128i_i64[0] - v3[44].m128i_i64[1]) >> 4);
  v29 = ((v3[46].m128i_i64[1] - v3[46].m128i_i64[0]) >> 3) + ((v3[45].m128i_i64[0] - v3[44].m128i_i64[1]) >> 4);
  if ( i != v9 )
  {
    v27 = 8 * i;
    v94 = 8 * i;
    if ( i < v9 )
    {
      i = v3->m128i_i64[0];
      v85 = v3->m128i_i64[0] + 8 * v9;
      if ( v85 != v3->m128i_i64[0] + v27 )
      {
        v86 = v3->m128i_i64[0] + v27;
        do
        {
          v87 = *(_QWORD *)(v85 - 8);
          v85 -= 8;
          if ( v87 )
            sub_B91220(v85);
        }
        while ( v86 != v85 );
        v4 = a1;
      }
      v3->m128i_i32[2] = v29;
    }
    else
    {
      v27 = v3->m128i_u32[3];
      if ( i > v27 )
      {
        v79 = (__int64 *)sub_C8D7D0(v3, &v3[1], i, 8, &v108);
        sub_A04E10((__int64)v3, v79, v80, v81, v82, v83);
        v84 = v108;
        if ( &v3[1] != (__m128i *)v3->m128i_i64[0] )
        {
          v92 = v108;
          _libc_free(v3->m128i_i64[0], v79);
          v84 = v92;
        }
        v3->m128i_i32[3] = v84;
        v9 = v3->m128i_u32[2];
        v3->m128i_i64[0] = (__int64)v79;
      }
      v30 = (_QWORD *)(v3->m128i_i64[0] + 8 * v9);
      for ( i = v94 + v3->m128i_i64[0]; (_QWORD *)i != v30; ++v30 )
      {
        if ( v30 )
          *v30 = 0;
      }
      v3->m128i_i32[2] = v29;
    }
  }
  sub_A11D10(&v108, v3, i, v27);
  if ( (v105 & 2) != 0 )
LABEL_114:
    sub_A05710(&v104.m128i_i8[8]);
  if ( (v105 & 1) != 0 && v104.m128i_i64[1] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v104.m128i_i64[1] + 8LL))(v104.m128i_i64[1]);
  v32 = v109 & 1;
  LOBYTE(v109) = v109 & 0xFD;
  if ( (_BYTE)v32 )
  {
LABEL_105:
    v15 = v108;
LABEL_12:
    *v4 = v15 | 1;
    goto LABEL_13;
  }
  LOBYTE(v105) = v105 & 0xFC;
  v104.m128i_i8[8] = v108;
  sub_A10370(v3, (__int64)&v114, (unsigned __int8)v108, v32, v31);
  v95 = (__int64 *)v3[48].m128i_i64[1];
  if ( (__int64 *)v3[48].m128i_i64[0] != v95 )
  {
    v90 = v4;
    v33 = (__int64 *)v3[48].m128i_i64[0];
    do
    {
      v34 = (_BYTE *)v33[1];
      if ( v34 && *v34 == 5 )
      {
        v35 = *(v34 - 16);
        if ( (v35 & 2) != 0 )
        {
          v36 = (_BYTE **)*((_QWORD *)v34 - 4);
          v37 = *((unsigned int *)v34 - 6);
        }
        else
        {
          v37 = (*((_WORD *)v34 - 8) >> 6) & 0xF;
          v36 = (_BYTE **)&v34[-8 * ((v35 >> 2) & 0xF) - 16];
        }
        v38 = &v36[v37];
        if ( v38 != v36 )
        {
          v39 = *v33;
          do
          {
            v40 = *v36;
            if ( *v36 && *v40 == 18 )
              sub_BA6610(v40, 5, v39);
            ++v36;
          }
          while ( v38 != v36 );
        }
      }
      v33 += 2;
    }
    while ( v95 != v33 );
    v4 = v90;
  }
  v41 = v3[48].m128i_i64[0];
  if ( v41 != v3[48].m128i_i64[1] )
    v3[48].m128i_i64[1] = v41;
  sub_A027C0((__int64)v3);
  sub_A12E50((__int64)v3);
  v43 = v3[15].m128i_i64[0];
  v44 = *(unsigned int *)(v43 + 72);
  if ( (_DWORD)v44 )
  {
    v45 = *(_DWORD *)(v43 + 32);
    if ( v45 > 0x1F )
    {
      *(_DWORD *)(v43 + 32) = 32;
      *(_QWORD *)(v43 + 24) >>= (unsigned __int8)v45 - 32;
    }
    else
    {
      *(_DWORD *)(v43 + 32) = 0;
    }
    v46 = *(_QWORD *)(v43 + 40);
    v42 = *(_QWORD *)(v43 + 48);
    v47 = *(_QWORD *)(v43 + 64) + 32 * v44 - 32;
    v91 = *(_QWORD *)(v43 + 56);
    v48 = v46;
    v96 = v42;
    *(_DWORD *)(v43 + 36) = *(_DWORD *)v47;
    *(_QWORD *)(v43 + 40) = *(_QWORD *)(v47 + 8);
    *(_QWORD *)(v43 + 48) = *(_QWORD *)(v47 + 16);
    *(_QWORD *)(v43 + 56) = *(_QWORD *)(v47 + 24);
    *(_QWORD *)(v47 + 8) = 0;
    *(_QWORD *)(v47 + 16) = 0;
    for ( *(_QWORD *)(v47 + 24) = 0; v96 != v48; v48 += 16 )
    {
      v49 = *(volatile signed __int32 **)(v48 + 8);
      if ( v49 )
      {
        if ( &_pthread_key_create )
        {
          v50 = _InterlockedExchangeAdd(v49 + 2, 0xFFFFFFFF);
        }
        else
        {
          v50 = *((_DWORD *)v49 + 2);
          v42 = (unsigned int)(v50 - 1);
          *((_DWORD *)v49 + 2) = v42;
        }
        if ( v50 == 1 )
        {
          v51 = *(void (**)(void))(*(_QWORD *)v49 + 16LL);
          if ( v51 != nullsub_25 )
            v51();
          if ( &_pthread_key_create )
          {
            v52 = _InterlockedExchangeAdd(v49 + 3, 0xFFFFFFFF);
          }
          else
          {
            v52 = *((_DWORD *)v49 + 3);
            *((_DWORD *)v49 + 3) = v52 - 1;
          }
          if ( v52 == 1 )
          {
            v53 = *(void (**)(void))(*(_QWORD *)v49 + 24LL);
            if ( (char *)v53 == (char *)sub_9C26E0 )
              (*(void (**)(void))(*(_QWORD *)v49 + 8LL))();
            else
              v53();
          }
        }
      }
    }
    if ( v46 )
      j_j___libc_free_0(v46, v91 - v46);
    v69 = (unsigned int)(*(_DWORD *)(v43 + 72) - 1);
    *(_DWORD *)(v43 + 72) = v69;
    v65 = (_QWORD *)(*(_QWORD *)(v43 + 64) + 32 * v69);
    v70 = v65[2];
    if ( v70 != v65[1] )
    {
      v98 = v65;
      v71 = v65[1];
      v72 = v3;
      v73 = v4;
      do
      {
        v74 = *(volatile signed __int32 **)(v71 + 8);
        if ( v74 )
        {
          v42 = (__int64)&_pthread_key_create;
          if ( &_pthread_key_create )
          {
            v75 = _InterlockedExchangeAdd(v74 + 2, 0xFFFFFFFF);
          }
          else
          {
            v75 = *((_DWORD *)v74 + 2);
            *((_DWORD *)v74 + 2) = v75 - 1;
          }
          if ( v75 == 1 )
          {
            v76 = *(void (**)())(*(_QWORD *)v74 + 16LL);
            if ( v76 != nullsub_25 )
            {
              ((void (__fastcall *)(volatile signed __int32 *))v76)(v74);
              v42 = (__int64)&_pthread_key_create;
            }
            if ( &_pthread_key_create )
            {
              v77 = _InterlockedExchangeAdd(v74 + 3, 0xFFFFFFFF);
            }
            else
            {
              v77 = *((_DWORD *)v74 + 3);
              v42 = (unsigned int)(v77 - 1);
              *((_DWORD *)v74 + 3) = v42;
            }
            if ( v77 == 1 )
            {
              v78 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v74 + 24LL);
              if ( v78 == sub_9C26E0 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v74 + 8LL))(v74);
              else
                v78((__int64)v74);
            }
          }
        }
        v71 += 16;
      }
      while ( v70 != v71 );
      v65 = v98;
      v4 = v73;
      v3 = v72;
    }
    v66 = v65[1];
    if ( v66 )
      j_j___libc_free_0(v66, v65[3] - v66);
  }
  m128i_i64 = (__int64)v3[23].m128i_i64;
  sub_9CDFE0(&v108, (__int64)v3[23].m128i_i64, 8 * v101 - v99, v42);
  if ( (v108 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *v4 = v108 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    m128i_i64 = v3[15].m128i_i64[0];
    sub_9CE5C0(v104.m128i_i64, m128i_i64, v24, v67);
    v68 = v104.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v104.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v104.m128i_i64[0] = 0;
      v108 = v68 | 1;
      sub_A049F0(&v108);
      sub_9C66B0(&v108);
      *v4 = 1;
      v108 = 0;
      sub_9C66B0(&v108);
      sub_9C66B0(v104.m128i_i64);
    }
    else
    {
      v104.m128i_i64[0] = 0;
      sub_9C66B0(v104.m128i_i64);
      *v4 = 1;
      v108 = 0;
      sub_9C66B0(&v108);
    }
  }
  if ( (v105 & 2) != 0 )
    goto LABEL_114;
  if ( (v105 & 1) != 0 )
  {
    v25 = v104.m128i_i64[1];
    if ( v104.m128i_i64[1] )
LABEL_28:
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v25 + 8LL))(v25, m128i_i64, v24);
  }
LABEL_13:
  v16 = &v104.m128i_i64[1];
  v104.m128i_i64[1] = v120;
  v105 = v121;
  v106 = v122;
  v107 = v123;
  v108 = v116;
  v109 = v117;
  v110 = v118;
  v111 = v119;
  sub_A01C60(&v108, &v104.m128i_i64[1]);
  v17 = v114;
  if ( v114 )
  {
    v18 = (__int64 *)v119;
    v19 = v123 + 8;
    if ( v123 + 8 > v119 )
    {
      do
      {
        v20 = *v18++;
        j_j___libc_free_0(v20, 512);
      }
      while ( v19 > (unsigned __int64)v18 );
      v17 = v114;
    }
    v16 = (__int64 *)(8 * v115);
    j_j___libc_free_0(v17, 8 * v115);
  }
  if ( (_BYTE *)v124 != v126 )
    _libc_free(v124, v16);
  return v4;
}
