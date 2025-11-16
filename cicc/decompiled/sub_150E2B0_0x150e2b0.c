// Function: sub_150E2B0
// Address: 0x150e2b0
//
__int64 __fastcall sub_150E2B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rcx
  __int64 v4; // r9
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // r15
  volatile signed __int32 *v19; // rbx
  signed __int32 v20; // ecx
  signed __int32 v21; // edx
  __int64 v22; // rbx
  __int64 v23; // r13
  volatile signed __int32 *v24; // rdi
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  unsigned __int64 v29; // r13
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r15
  volatile signed __int32 *v33; // r14
  signed __int32 v34; // eax
  signed __int32 v35; // eax
  __int64 v36; // rbx
  __int64 v37; // r14
  volatile signed __int32 *v38; // r15
  signed __int32 v39; // eax
  signed __int32 v40; // eax
  _QWORD *v41; // r14
  __int64 v42; // rbx
  __int64 v43; // r13
  __int64 v44; // rdi
  _QWORD *v45; // rdi
  __int64 v46; // rbx
  __int64 v47; // r13
  volatile signed __int32 *v48; // r15
  signed __int32 v49; // eax
  signed __int32 v50; // eax
  unsigned __int64 *v51; // rbx
  unsigned __int64 *v52; // r14
  unsigned __int64 v53; // rdi
  unsigned __int64 *v54; // rbx
  unsigned __int64 v55; // r14
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  __int64 v58; // rax
  unsigned __int64 v59; // r8
  __int64 v60; // r14
  __int64 v61; // rbx
  unsigned __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // r8
  unsigned __int64 v66; // r13
  __int64 v67; // rdi
  __int64 v68; // r15
  __int64 v69; // r14
  volatile signed __int32 *v70; // r12
  signed __int32 v71; // eax
  signed __int32 v72; // eax
  __int64 v73; // rbx
  __int64 v74; // r12
  volatile signed __int32 *v75; // r14
  signed __int32 v76; // eax
  signed __int32 v77; // eax
  unsigned __int64 *v79; // r9
  unsigned int v80; // edx
  __int64 v81; // r10
  unsigned int v82; // r11d
  __int64 v83; // rax
  __int64 v84; // rdx
  unsigned __int64 v85; // r8
  __int64 v86; // rdi
  char v87; // cl
  _BYTE *v89; // [rsp+30h] [rbp-570h]
  unsigned __int64 v90; // [rsp+30h] [rbp-570h]
  _QWORD *v91; // [rsp+30h] [rbp-570h]
  __m128i v92; // [rsp+70h] [rbp-530h] BYREF
  __m128i v93; // [rsp+80h] [rbp-520h] BYREF
  __int64 v94; // [rsp+90h] [rbp-510h]
  __int64 v95; // [rsp+98h] [rbp-508h]
  __int64 v96; // [rsp+A0h] [rbp-500h]
  __int64 v97; // [rsp+A8h] [rbp-4F8h]
  _BYTE *v98; // [rsp+B0h] [rbp-4F0h] BYREF
  __int64 v99; // [rsp+B8h] [rbp-4E8h]
  _BYTE v100[256]; // [rsp+C0h] [rbp-4E0h] BYREF
  __int64 v101; // [rsp+1C0h] [rbp-3E0h]
  __m128i v102[2]; // [rsp+1D0h] [rbp-3D0h] BYREF
  __int64 v103; // [rsp+1F0h] [rbp-3B0h]
  __int64 v104; // [rsp+1F8h] [rbp-3A8h]
  __int64 v105; // [rsp+200h] [rbp-3A0h]
  __int64 v106; // [rsp+208h] [rbp-398h]
  _BYTE *v107; // [rsp+210h] [rbp-390h] BYREF
  __int64 v108; // [rsp+218h] [rbp-388h]
  _BYTE v109[256]; // [rsp+220h] [rbp-380h] BYREF
  __int64 v110; // [rsp+320h] [rbp-280h]
  _QWORD *v111; // [rsp+330h] [rbp-270h] BYREF
  _QWORD *v112; // [rsp+338h] [rbp-268h]
  __int64 v113; // [rsp+340h] [rbp-260h]
  __int64 v114; // [rsp+370h] [rbp-230h]
  __int64 v115; // [rsp+378h] [rbp-228h]
  __int64 v116; // [rsp+380h] [rbp-220h]
  unsigned __int64 v117; // [rsp+388h] [rbp-218h]
  unsigned int v118; // [rsp+390h] [rbp-210h]
  char v119; // [rsp+398h] [rbp-208h] BYREF
  _QWORD *v120; // [rsp+4B8h] [rbp-E8h]
  _QWORD v121[13]; // [rsp+4C8h] [rbp-D8h] BYREF
  __int64 *v122; // [rsp+530h] [rbp-70h]
  __int64 v123; // [rsp+540h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a2 + 56);
  v4 = *(_QWORD *)a2;
  v94 = 0x200000000LL;
  v5 = *(_QWORD *)(a2 + 8);
  v98 = v100;
  v99 = 0x800000000LL;
  v92.m128i_i64[0] = v4;
  v6 = (v3 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
  v92.m128i_i64[1] = v5;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v101 = 0;
  v93 = (__m128i)v6;
  v7 = v3 & 0x3F;
  if ( (v3 & 0x3F) != 0 )
  {
    if ( v5 <= v6 )
      goto LABEL_185;
    v79 = (unsigned __int64 *)(v6 + v4);
    if ( v5 >= v6 + 8 )
    {
      v85 = *v79;
      v93.m128i_i64[0] = v6 + 8;
      v82 = 64;
    }
    else
    {
      v80 = v5 - v6;
      v81 = v80;
      v82 = 8 * v80;
      v83 = v80 + v6;
      if ( !v80 )
      {
        v93.m128i_i64[0] = v83;
        LODWORD(v94) = 0;
        goto LABEL_185;
      }
      v84 = 0;
      v85 = 0;
      do
      {
        v86 = *((unsigned __int8 *)v79 + v84);
        v87 = 8 * v84++;
        v85 |= v86 << v87;
        v93.m128i_i64[1] = v85;
      }
      while ( v81 != v84 );
      v93.m128i_i64[0] = v83;
      LODWORD(v94) = v82;
      if ( v7 > v82 )
LABEL_185:
        sub_16BD130("Unexpected end of file", 1);
    }
    LODWORD(v94) = v82 - v7;
    v93.m128i_i64[1] = v85 >> v7;
  }
  v8 = sub_22077B0(392);
  v9 = v8;
  if ( v8 )
  {
    *(_DWORD *)(v8 + 8) = 0;
    v10 = v8 + 8;
    *(_QWORD *)(v10 + 8) = 0;
    *(_QWORD *)(v9 + 24) = v10;
    *(_QWORD *)(v9 + 32) = v10;
    *(_QWORD *)(v9 + 64) = 0x2800000000LL;
    *(_QWORD *)(v9 + 104) = v9 + 88;
    *(_QWORD *)(v9 + 112) = v9 + 88;
    *(_QWORD *)(v9 + 152) = v9 + 136;
    *(_QWORD *)(v9 + 160) = v9 + 136;
    *(_QWORD *)(v9 + 208) = v9 + 192;
    *(_QWORD *)(v9 + 216) = v9 + 192;
    *(_QWORD *)(v9 + 256) = v9 + 240;
    *(_QWORD *)(v9 + 264) = v9 + 240;
    *(_QWORD *)(v9 + 40) = 0;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = 0;
    *(_DWORD *)(v9 + 88) = 0;
    *(_QWORD *)(v9 + 96) = 0;
    *(_QWORD *)(v9 + 120) = 0;
    *(_DWORD *)(v9 + 136) = 0;
    *(_QWORD *)(v9 + 144) = 0;
    *(_QWORD *)(v9 + 168) = 0;
    *(_WORD *)(v9 + 176) = 0;
    *(_BYTE *)(v9 + 178) = 0;
    *(_DWORD *)(v9 + 192) = 0;
    *(_QWORD *)(v9 + 200) = 0;
    *(_QWORD *)(v9 + 224) = 0;
    *(_DWORD *)(v9 + 240) = 0;
    *(_QWORD *)(v9 + 248) = 0;
    *(_QWORD *)(v9 + 272) = 0;
    *(_QWORD *)(v9 + 280) = 0;
    *(_QWORD *)(v9 + 288) = 0;
    *(_QWORD *)(v9 + 296) = v9 + 312;
    *(_QWORD *)(v9 + 304) = 0x400000000LL;
    *(_QWORD *)(v9 + 344) = v9 + 360;
    *(_QWORD *)(v9 + 352) = 0;
    *(_QWORD *)(v9 + 360) = 0;
    *(_QWORD *)(v9 + 368) = 1;
    *(_QWORD *)(v9 + 384) = v9 + 280;
  }
  v11 = _mm_loadu_si128(&v92);
  v12 = _mm_loadu_si128(&v93);
  v103 = v94;
  v104 = v95;
  v95 = 0;
  v105 = v96;
  v96 = 0;
  v106 = v97;
  v107 = v109;
  v97 = 0;
  v108 = 0x800000000LL;
  v102[0] = v11;
  v102[1] = v12;
  if ( (_DWORD)v99 )
    sub_14F2DD0((__int64)&v107, (__int64 *)&v98);
  v13 = *(_QWORD *)(a2 + 40);
  v110 = v101;
  sub_14F3D20((__int64)&v111, v102, *(_QWORD *)(a2 + 32), v13, v9, 0, *(_OWORD *)(a2 + 16));
  v14 = 32LL * (unsigned int)v108;
  v89 = v107;
  v15 = (unsigned __int64)&v107[v14];
  if ( v107 != &v107[v14] )
  {
    do
    {
      v16 = *(_QWORD *)(v15 - 24);
      v17 = *(_QWORD *)(v15 - 16);
      v15 -= 32LL;
      v18 = v16;
      if ( v17 != v16 )
      {
        do
        {
          while ( 1 )
          {
            v19 = *(volatile signed __int32 **)(v18 + 8);
            if ( v19 )
            {
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v19 + 2);
                *((_DWORD *)v19 + 2) = v20 - 1;
              }
              if ( v20 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 16LL))(v19);
                if ( &_pthread_key_create )
                {
                  v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v21 = *((_DWORD *)v19 + 3);
                  *((_DWORD *)v19 + 3) = v21 - 1;
                }
                if ( v21 == 1 )
                  break;
              }
            }
            v18 += 16;
            if ( v17 == v18 )
              goto LABEL_18;
          }
          v18 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
        }
        while ( v17 != v18 );
LABEL_18:
        v16 = *(_QWORD *)(v15 + 8);
      }
      if ( v16 )
        j_j___libc_free_0(v16, *(_QWORD *)(v15 + 24) - v16);
    }
    while ( v89 != (_BYTE *)v15 );
    v15 = (unsigned __int64)v107;
  }
  if ( (_BYTE *)v15 != v109 )
    _libc_free(v15);
  v22 = v105;
  v23 = v104;
  if ( v105 != v104 )
  {
    do
    {
      while ( 1 )
      {
        v24 = *(volatile signed __int32 **)(v23 + 8);
        if ( v24 )
        {
          if ( &_pthread_key_create )
          {
            v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
          }
          else
          {
            v25 = *((_DWORD *)v24 + 2);
            *((_DWORD *)v24 + 2) = v25 - 1;
          }
          if ( v25 == 1 )
          {
            (*(void (**)(void))(*(_QWORD *)v24 + 16LL))();
            if ( &_pthread_key_create )
            {
              v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
            }
            else
            {
              v26 = *((_DWORD *)v24 + 3);
              *((_DWORD *)v24 + 3) = v26 - 1;
            }
            if ( v26 == 1 )
              break;
          }
        }
        v23 += 16;
        if ( v22 == v23 )
          goto LABEL_36;
      }
      v23 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
    }
    while ( v22 != v23 );
LABEL_36:
    v23 = v104;
  }
  if ( v23 )
    j_j___libc_free_0(v23, v106 - v23);
  sub_150B5F0(v102[0].m128i_i64, (__int64)&v111);
  v27 = v102[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v102[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v27;
  }
  else
  {
    v63 = *(unsigned __int8 *)(a1 + 8);
    *(_QWORD *)a1 = v9;
    v9 = 0;
    *(_BYTE *)(a1 + 8) = v63 & 0xFC | 2;
  }
  if ( v122 != &v123 )
    j_j___libc_free_0(v122, v123 + 1);
  j___libc_free_0(v121[10]);
  j___libc_free_0(v121[6]);
  if ( v120 != v121 )
    j_j___libc_free_0(v120, v121[0] + 1LL);
  v28 = 32LL * v118;
  v90 = v117;
  v29 = v117 + v28;
  if ( v117 != v117 + v28 )
  {
    do
    {
      v30 = *(_QWORD *)(v29 - 24);
      v31 = *(_QWORD *)(v29 - 16);
      v29 -= 32LL;
      v32 = v30;
      if ( v31 != v30 )
      {
        do
        {
          while ( 1 )
          {
            v33 = *(volatile signed __int32 **)(v32 + 8);
            if ( v33 )
            {
              if ( &_pthread_key_create )
              {
                v34 = _InterlockedExchangeAdd(v33 + 2, 0xFFFFFFFF);
              }
              else
              {
                v34 = *((_DWORD *)v33 + 2);
                *((_DWORD *)v33 + 2) = v34 - 1;
              }
              if ( v34 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 16LL))(v33);
                if ( &_pthread_key_create )
                {
                  v35 = _InterlockedExchangeAdd(v33 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v35 = *((_DWORD *)v33 + 3);
                  *((_DWORD *)v33 + 3) = v35 - 1;
                }
                if ( v35 == 1 )
                  break;
              }
            }
            v32 += 16;
            if ( v31 == v32 )
              goto LABEL_57;
          }
          v32 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 24LL))(v33);
        }
        while ( v31 != v32 );
LABEL_57:
        v30 = *(_QWORD *)(v29 + 8);
      }
      if ( v30 )
        j_j___libc_free_0(v30, *(_QWORD *)(v29 + 24) - v30);
    }
    while ( v90 != v29 );
    v29 = v117;
  }
  if ( (char *)v29 != &v119 )
    _libc_free(v29);
  v36 = v115;
  v37 = v114;
  if ( v115 != v114 )
  {
    do
    {
      while ( 1 )
      {
        v38 = *(volatile signed __int32 **)(v37 + 8);
        if ( v38 )
        {
          if ( &_pthread_key_create )
          {
            v39 = _InterlockedExchangeAdd(v38 + 2, 0xFFFFFFFF);
          }
          else
          {
            v39 = *((_DWORD *)v38 + 2);
            *((_DWORD *)v38 + 2) = v39 - 1;
          }
          if ( v39 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v38 + 16LL))(v38);
            if ( &_pthread_key_create )
            {
              v40 = _InterlockedExchangeAdd(v38 + 3, 0xFFFFFFFF);
            }
            else
            {
              v40 = *((_DWORD *)v38 + 3);
              *((_DWORD *)v38 + 3) = v40 - 1;
            }
            if ( v40 == 1 )
              break;
          }
        }
        v37 += 16;
        if ( v36 == v37 )
          goto LABEL_75;
      }
      v37 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v38 + 24LL))(v38);
    }
    while ( v36 != v37 );
LABEL_75:
    v37 = v114;
  }
  if ( v37 )
    j_j___libc_free_0(v37, v116 - v37);
  v41 = v111;
  v91 = v112;
  if ( v112 != v111 )
  {
    do
    {
      v42 = v41[9];
      v43 = v41[8];
      if ( v42 != v43 )
      {
        do
        {
          v44 = *(_QWORD *)(v43 + 8);
          if ( v44 != v43 + 24 )
            j_j___libc_free_0(v44, *(_QWORD *)(v43 + 24) + 1LL);
          v43 += 40;
        }
        while ( v42 != v43 );
        v43 = v41[8];
      }
      if ( v43 )
        j_j___libc_free_0(v43, v41[10] - v43);
      v45 = (_QWORD *)v41[4];
      if ( v45 != v41 + 6 )
        j_j___libc_free_0(v45, v41[6] + 1LL);
      v46 = v41[2];
      v47 = v41[1];
      if ( v46 != v47 )
      {
        do
        {
          while ( 1 )
          {
            v48 = *(volatile signed __int32 **)(v47 + 8);
            if ( v48 )
            {
              if ( &_pthread_key_create )
              {
                v49 = _InterlockedExchangeAdd(v48 + 2, 0xFFFFFFFF);
              }
              else
              {
                v49 = *((_DWORD *)v48 + 2);
                *((_DWORD *)v48 + 2) = v49 - 1;
              }
              if ( v49 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v48 + 16LL))(v48);
                if ( &_pthread_key_create )
                {
                  v50 = _InterlockedExchangeAdd(v48 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v50 = *((_DWORD *)v48 + 3);
                  *((_DWORD *)v48 + 3) = v50 - 1;
                }
                if ( v50 == 1 )
                  break;
              }
            }
            v47 += 16;
            if ( v46 == v47 )
              goto LABEL_99;
          }
          v47 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v48 + 24LL))(v48);
        }
        while ( v46 != v47 );
LABEL_99:
        v47 = v41[1];
      }
      if ( v47 )
        j_j___libc_free_0(v47, v41[3] - v47);
      v41 += 11;
    }
    while ( v91 != v41 );
    v41 = v111;
  }
  if ( v41 )
    j_j___libc_free_0(v41, v113 - (_QWORD)v41);
  if ( v9 )
  {
    v51 = *(unsigned __int64 **)(v9 + 296);
    v52 = &v51[*(unsigned int *)(v9 + 304)];
    while ( v52 != v51 )
    {
      v53 = *v51++;
      _libc_free(v53);
    }
    v54 = *(unsigned __int64 **)(v9 + 344);
    v55 = (unsigned __int64)&v54[2 * *(unsigned int *)(v9 + 352)];
    if ( v54 != (unsigned __int64 *)v55 )
    {
      do
      {
        v56 = *v54;
        v54 += 2;
        _libc_free(v56);
      }
      while ( (unsigned __int64 *)v55 != v54 );
      v55 = *(_QWORD *)(v9 + 344);
    }
    if ( v55 != v9 + 360 )
      _libc_free(v55);
    v57 = *(_QWORD *)(v9 + 296);
    if ( v57 != v9 + 312 )
      _libc_free(v57);
    sub_14EAEF0(*(_QWORD **)(v9 + 248));
    sub_14EAEF0(*(_QWORD **)(v9 + 200));
    sub_14EA590(*(_QWORD *)(v9 + 144));
    sub_14EADF0(*(_QWORD **)(v9 + 96));
    if ( *(_DWORD *)(v9 + 60) )
    {
      v58 = *(unsigned int *)(v9 + 56);
      v59 = *(_QWORD *)(v9 + 48);
      if ( (_DWORD)v58 )
      {
        v60 = 8 * v58;
        v61 = 0;
        do
        {
          v62 = *(_QWORD *)(v59 + v61);
          if ( v62 != -8 && v62 )
          {
            _libc_free(v62);
            v59 = *(_QWORD *)(v9 + 48);
          }
          v61 += 8;
        }
        while ( v61 != v60 );
      }
    }
    else
    {
      v59 = *(_QWORD *)(v9 + 48);
    }
    _libc_free(v59);
    sub_14EA160(*(_QWORD **)(v9 + 16));
    j_j___libc_free_0(v9, 392);
  }
  v64 = (__int64)v98;
  v65 = 32LL * (unsigned int)v99;
  v66 = (unsigned __int64)&v98[v65];
  if ( v98 != &v98[v65] )
  {
    do
    {
      v67 = *(_QWORD *)(v66 - 24);
      v68 = *(_QWORD *)(v66 - 16);
      v66 -= 32LL;
      v69 = v67;
      if ( v68 != v67 )
      {
        do
        {
          while ( 1 )
          {
            v70 = *(volatile signed __int32 **)(v69 + 8);
            if ( v70 )
            {
              if ( &_pthread_key_create )
              {
                v71 = _InterlockedExchangeAdd(v70 + 2, 0xFFFFFFFF);
              }
              else
              {
                v71 = *((_DWORD *)v70 + 2);
                *((_DWORD *)v70 + 2) = v71 - 1;
              }
              if ( v71 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v70 + 16LL))(v70);
                if ( &_pthread_key_create )
                {
                  v72 = _InterlockedExchangeAdd(v70 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v72 = *((_DWORD *)v70 + 3);
                  *((_DWORD *)v70 + 3) = v72 - 1;
                }
                if ( v72 == 1 )
                  break;
              }
            }
            v69 += 16;
            if ( v68 == v69 )
              goto LABEL_147;
          }
          v69 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v70 + 24LL))(v70);
        }
        while ( v68 != v69 );
LABEL_147:
        v67 = *(_QWORD *)(v66 + 8);
      }
      if ( v67 )
        j_j___libc_free_0(v67, *(_QWORD *)(v66 + 24) - v67);
    }
    while ( v64 != v66 );
    v66 = (unsigned __int64)v98;
  }
  if ( (_BYTE *)v66 != v100 )
    _libc_free(v66);
  v73 = v96;
  v74 = v95;
  if ( v96 != v95 )
  {
    do
    {
      while ( 1 )
      {
        v75 = *(volatile signed __int32 **)(v74 + 8);
        if ( v75 )
        {
          if ( &_pthread_key_create )
          {
            v76 = _InterlockedExchangeAdd(v75 + 2, 0xFFFFFFFF);
          }
          else
          {
            v76 = *((_DWORD *)v75 + 2);
            *((_DWORD *)v75 + 2) = v76 - 1;
          }
          if ( v76 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v75 + 16LL))(v75);
            if ( &_pthread_key_create )
            {
              v77 = _InterlockedExchangeAdd(v75 + 3, 0xFFFFFFFF);
            }
            else
            {
              v77 = *((_DWORD *)v75 + 3);
              *((_DWORD *)v75 + 3) = v77 - 1;
            }
            if ( v77 == 1 )
              break;
          }
        }
        v74 += 16;
        if ( v73 == v74 )
          goto LABEL_165;
      }
      v74 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v75 + 24LL))(v75);
    }
    while ( v73 != v74 );
LABEL_165:
    v74 = v95;
  }
  if ( v74 )
    j_j___libc_free_0(v74, v97 - v74);
  return a1;
}
