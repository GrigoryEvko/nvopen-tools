// Function: sub_9F0EA0
// Address: 0x9f0ea0
//
__int64 __fastcall sub_9F0EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __m128i v13; // xmm1
  __m128i v14; // xmm0
  __int64 v15; // rdx
  __m128i *v16; // rsi
  __int64 v17; // r8
  _BYTE *v18; // r15
  _BYTE *v19; // r12
  _BYTE *v20; // r15
  __int64 v21; // rdi
  __int64 v22; // r14
  __int64 v23; // rbx
  volatile signed __int32 *v24; // r13
  signed __int32 v25; // eax
  void (*v26)(); // rax
  signed __int32 v27; // eax
  __int64 (__fastcall *v28)(__int64); // rdx
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // r12
  volatile signed __int32 *v32; // r13
  signed __int32 v33; // eax
  void (*v34)(); // rax
  signed __int32 v35; // eax
  __int64 (__fastcall *v36)(__int64); // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r8
  char *v41; // r15
  _BYTE *v42; // r12
  __int64 v43; // rdi
  __int64 v44; // r14
  __int64 v45; // rbx
  volatile signed __int32 *v46; // r13
  signed __int32 v47; // eax
  void (*v48)(); // rax
  signed __int32 v49; // eax
  __int64 v50; // rbx
  __int64 v51; // r14
  volatile signed __int32 *v52; // r13
  signed __int32 v53; // eax
  void (*v54)(); // rax
  signed __int32 v55; // eax
  __int64 (__fastcall *v56)(__int64); // rdx
  _QWORD *v57; // r15
  _QWORD *v58; // rbx
  __int64 v59; // r14
  __int64 v60; // r13
  __int64 v61; // rdi
  _QWORD *v62; // rdi
  __int64 v63; // r12
  __int64 v64; // r13
  volatile signed __int32 *v65; // r14
  signed __int32 v66; // eax
  void (*v67)(); // rax
  signed __int32 v68; // eax
  __int64 (__fastcall *v69)(__int64); // rdx
  __int64 v70; // r8
  _BYTE *v71; // r15
  _BYTE *v72; // r14
  __int64 v73; // rdi
  __int64 v74; // r15
  __int64 v75; // rbx
  volatile signed __int32 *v76; // r12
  signed __int32 v77; // eax
  void (*v78)(); // rax
  signed __int32 v79; // eax
  __int64 (__fastcall *v80)(__int64); // rcx
  __int64 v81; // rbx
  __int64 v82; // r12
  volatile signed __int32 *v83; // r13
  signed __int32 v84; // eax
  void (*v85)(); // rax
  signed __int32 v86; // eax
  __int64 (__fastcall *v87)(__int64); // rdx
  __int64 v89; // [rsp+8h] [rbp-5F8h]
  __int64 v90; // [rsp+20h] [rbp-5E0h]
  __int64 v91; // [rsp+20h] [rbp-5E0h]
  _BYTE *v93; // [rsp+38h] [rbp-5C8h]
  _BYTE *v94; // [rsp+38h] [rbp-5C8h]
  _BYTE v95[16]; // [rsp+70h] [rbp-590h] BYREF
  void (__fastcall *v96)(_BYTE *, _BYTE *, __int64); // [rsp+80h] [rbp-580h]
  __m128i v97; // [rsp+90h] [rbp-570h] BYREF
  __m128i v98; // [rsp+A0h] [rbp-560h] BYREF
  __int64 v99; // [rsp+B0h] [rbp-550h]
  __int64 v100; // [rsp+B8h] [rbp-548h]
  __int64 v101; // [rsp+C0h] [rbp-540h]
  __int64 v102; // [rsp+C8h] [rbp-538h]
  _BYTE *v103; // [rsp+D0h] [rbp-530h] BYREF
  __int64 v104; // [rsp+D8h] [rbp-528h]
  _BYTE v105[256]; // [rsp+E0h] [rbp-520h] BYREF
  __int64 v106; // [rsp+1E0h] [rbp-420h]
  __m128i v107[2]; // [rsp+1F0h] [rbp-410h] BYREF
  __int64 v108; // [rsp+210h] [rbp-3F0h]
  __int64 v109; // [rsp+218h] [rbp-3E8h]
  __int64 v110; // [rsp+220h] [rbp-3E0h]
  __int64 v111; // [rsp+228h] [rbp-3D8h]
  _BYTE *v112; // [rsp+230h] [rbp-3D0h] BYREF
  __int64 v113; // [rsp+238h] [rbp-3C8h]
  _BYTE v114[256]; // [rsp+240h] [rbp-3C0h] BYREF
  __int64 v115; // [rsp+340h] [rbp-2C0h]
  __int64 v116; // [rsp+350h] [rbp-2B0h] BYREF
  _QWORD *v117; // [rsp+358h] [rbp-2A8h]
  __int64 v118; // [rsp+360h] [rbp-2A0h]
  __int64 v119; // [rsp+390h] [rbp-270h]
  __int64 v120; // [rsp+398h] [rbp-268h]
  __int64 v121; // [rsp+3A0h] [rbp-260h]
  _BYTE *v122; // [rsp+3A8h] [rbp-258h]
  unsigned int v123; // [rsp+3B0h] [rbp-250h]
  char v124; // [rsp+3B8h] [rbp-248h] BYREF
  _QWORD *v125; // [rsp+4D8h] [rbp-128h]
  _QWORD v126[8]; // [rsp+4E8h] [rbp-118h] BYREF
  unsigned int v127; // [rsp+528h] [rbp-D8h]
  __int64 v128; // [rsp+538h] [rbp-C8h]
  unsigned int v129; // [rsp+548h] [rbp-B8h]
  __int64 *v130; // [rsp+550h] [rbp-B0h]
  __int64 v131; // [rsp+560h] [rbp-A0h] BYREF
  _BYTE v132[16]; // [rsp+580h] [rbp-80h] BYREF
  void (__fastcall *v133)(_BYTE *, _BYTE *, __int64); // [rsp+590h] [rbp-70h]
  __int64 v134; // [rsp+5A0h] [rbp-60h]
  __int64 v135; // [rsp+5B0h] [rbp-50h]
  __int64 v136; // [rsp+5B8h] [rbp-48h]
  __int64 v137; // [rsp+5C8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)a2;
  v7 = (__int64)&v97;
  v98 = 0u;
  v97.m128i_i64[1] = v5;
  v97.m128i_i64[0] = v6;
  v8 = *(_QWORD *)(a2 + 56);
  v99 = 0x200000000LL;
  v103 = v105;
  v104 = 0x800000000LL;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v106 = 0;
  sub_9CDFE0(&v116, (__int64)&v97, v8, a4);
  v9 = v116 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v116 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v10 = sub_22077B0(584);
    v11 = v10;
    if ( v10 )
    {
      *(_DWORD *)(v10 + 8) = 0;
      v12 = v10 + 8;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v11 + 24) = v12;
      *(_QWORD *)(v11 + 32) = v12;
      *(_QWORD *)(v11 + 136) = v11 + 152;
      *(_QWORD *)(v11 + 64) = 0x2000000000LL;
      *(_QWORD *)(v11 + 168) = v11 + 72;
      *(_QWORD *)(v11 + 88) = v11 + 104;
      *(_QWORD *)(v11 + 232) = v11 + 216;
      *(_QWORD *)(v11 + 240) = v11 + 216;
      *(_QWORD *)(v11 + 96) = 0x400000000LL;
      *(_QWORD *)(v11 + 280) = v11 + 264;
      *(_QWORD *)(v11 + 288) = v11 + 264;
      *(_QWORD *)(v11 + 40) = 0;
      *(_QWORD *)(v11 + 48) = 0;
      *(_QWORD *)(v11 + 56) = 0;
      *(_QWORD *)(v11 + 72) = 0;
      *(_QWORD *)(v11 + 80) = 0;
      *(_QWORD *)(v11 + 144) = 0;
      *(_QWORD *)(v11 + 152) = 0;
      *(_QWORD *)(v11 + 160) = 1;
      *(_QWORD *)(v11 + 176) = 0;
      *(_QWORD *)(v11 + 184) = 0;
      *(_QWORD *)(v11 + 192) = 0;
      *(_DWORD *)(v11 + 200) = 0;
      *(_DWORD *)(v11 + 216) = 0;
      *(_QWORD *)(v11 + 224) = 0;
      *(_QWORD *)(v11 + 248) = 0;
      *(_DWORD *)(v11 + 264) = 0;
      *(_QWORD *)(v11 + 272) = 0;
      *(_QWORD *)(v11 + 296) = 0;
      *(_QWORD *)(v11 + 304) = 0;
      *(_QWORD *)(v11 + 312) = 0;
      *(_QWORD *)(v11 + 440) = 0x400000000LL;
      *(_QWORD *)(v11 + 480) = v11 + 496;
      *(_QWORD *)(v11 + 320) = 0;
      *(_DWORD *)(v11 + 328) = 0;
      *(_QWORD *)(v11 + 336) = 0;
      *(_DWORD *)(v11 + 344) = 0;
      *(_QWORD *)(v11 + 352) = 0;
      *(_QWORD *)(v11 + 360) = 0;
      *(_QWORD *)(v11 + 368) = 0;
      *(_DWORD *)(v11 + 376) = 0;
      *(_QWORD *)(v11 + 384) = 0;
      *(_QWORD *)(v11 + 392) = 0;
      *(_QWORD *)(v11 + 400) = 0;
      *(_DWORD *)(v11 + 408) = 0;
      *(_QWORD *)(v11 + 416) = 0;
      *(_QWORD *)(v11 + 424) = 0;
      *(_QWORD *)(v11 + 432) = v11 + 448;
      *(_QWORD *)(v11 + 488) = 0;
      *(_QWORD *)(v11 + 496) = 0;
      *(_QWORD *)(v11 + 504) = 1;
      *(_QWORD *)(v11 + 512) = v11 + 416;
      *(_QWORD *)(v11 + 520) = 0;
      *(_QWORD *)(v11 + 528) = 0;
      *(_QWORD *)(v11 + 536) = 0;
      *(_QWORD *)(v11 + 544) = 0;
      *(_QWORD *)(v11 + 552) = 0;
      *(_QWORD *)(v11 + 560) = 0;
      *(_QWORD *)(v11 + 568) = 0;
      *(_DWORD *)(v11 + 576) = 0;
    }
    v13 = _mm_loadu_si128(&v97);
    v96 = 0;
    v14 = _mm_loadu_si128(&v98);
    v108 = v99;
    v109 = v100;
    v100 = 0;
    v110 = v101;
    v101 = 0;
    v111 = v102;
    v112 = v114;
    v102 = 0;
    v113 = 0x800000000LL;
    v107[0] = v13;
    v107[1] = v14;
    if ( (_DWORD)v104 )
      sub_9D06B0((__int64)&v112, (__int64)&v103);
    v15 = *(_QWORD *)(a2 + 40);
    v115 = v106;
    v16 = v107;
    sub_9D1140((__int64)&v116, v107, *(_QWORD *)(a2 + 32), v15, v11, (__int64)v95, *(_OWORD *)(a2 + 16));
    v17 = 32LL * (unsigned int)v113;
    v18 = &v112[v17];
    if ( v112 != &v112[v17] )
    {
      v89 = v11;
      v19 = &v112[v17];
      v20 = v112;
      while ( 1 )
      {
        v21 = *((_QWORD *)v19 - 3);
        v22 = *((_QWORD *)v19 - 2);
        v19 -= 32;
        v23 = v21;
        if ( v22 == v21 )
          goto LABEL_23;
        do
        {
          while ( 1 )
          {
            v24 = *(volatile signed __int32 **)(v23 + 8);
            if ( !v24 )
              goto LABEL_10;
            if ( &_pthread_key_create )
            {
              v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
            }
            else
            {
              v25 = *((_DWORD *)v24 + 2);
              *((_DWORD *)v24 + 2) = v25 - 1;
            }
            if ( v25 != 1 )
              goto LABEL_10;
            v26 = *(void (**)())(*(_QWORD *)v24 + 16LL);
            if ( v26 != nullsub_25 )
              ((void (__fastcall *)(volatile signed __int32 *))v26)(v24);
            if ( &_pthread_key_create )
            {
              v27 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
            }
            else
            {
              v27 = *((_DWORD *)v24 + 3);
              *((_DWORD *)v24 + 3) = v27 - 1;
            }
            if ( v27 != 1 )
              goto LABEL_10;
            v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 24LL);
            if ( v28 == sub_9C26E0 )
              break;
            v28((__int64)v24);
LABEL_10:
            v23 += 16;
            if ( v22 == v23 )
              goto LABEL_22;
          }
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 8LL))(v24);
          v23 += 16;
        }
        while ( v22 != v23 );
LABEL_22:
        v21 = *((_QWORD *)v19 + 1);
LABEL_23:
        if ( v21 )
        {
          v16 = (__m128i *)(*((_QWORD *)v19 + 3) - v21);
          j_j___libc_free_0(v21, v16);
        }
        if ( v20 == v19 )
        {
          v11 = v89;
          v18 = v112;
          break;
        }
      }
    }
    if ( v18 != v114 )
      _libc_free(v18, v16);
    v29 = v109;
    if ( v110 != v109 )
    {
      v30 = v11;
      v31 = v110;
      while ( 1 )
      {
        v32 = *(volatile signed __int32 **)(v29 + 8);
        if ( !v32 )
          goto LABEL_31;
        if ( &_pthread_key_create )
        {
          v33 = _InterlockedExchangeAdd(v32 + 2, 0xFFFFFFFF);
        }
        else
        {
          v33 = *((_DWORD *)v32 + 2);
          *((_DWORD *)v32 + 2) = v33 - 1;
        }
        if ( v33 != 1 )
          goto LABEL_31;
        v34 = *(void (**)())(*(_QWORD *)v32 + 16LL);
        if ( v34 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v34)(v32);
        if ( &_pthread_key_create )
        {
          v35 = _InterlockedExchangeAdd(v32 + 3, 0xFFFFFFFF);
        }
        else
        {
          v35 = *((_DWORD *)v32 + 3);
          *((_DWORD *)v32 + 3) = v35 - 1;
        }
        if ( v35 != 1 )
          goto LABEL_31;
        v36 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v32 + 24LL);
        if ( v36 == sub_9C26E0 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v32 + 8LL))(v32);
          v29 += 16;
          if ( v31 == v29 )
          {
LABEL_43:
            v11 = v30;
            v29 = v109;
            break;
          }
        }
        else
        {
          v36((__int64)v32);
LABEL_31:
          v29 += 16;
          if ( v31 == v29 )
            goto LABEL_43;
        }
      }
    }
    if ( v29 )
      j_j___libc_free_0(v29, v111 - v29);
    if ( v96 )
      v96(v95, v95, 3);
    sub_9EF580(v107[0].m128i_i64, (__int64)&v116);
    v37 = v107[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v107[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v39 = v136;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v37;
      if ( !v39 )
        goto LABEL_51;
    }
    else
    {
      v38 = *(unsigned __int8 *)(a1 + 8);
      v39 = v136;
      *(_QWORD *)a1 = v11;
      v11 = 0;
      *(_BYTE *)(a1 + 8) = v38 & 0xFC | 2;
      if ( !v39 )
      {
LABEL_51:
        if ( v134 )
          j_j___libc_free_0(v134, v135 - v134);
        if ( v133 )
          v133(v132, v132, 3);
        if ( v130 != &v131 )
          j_j___libc_free_0(v130, v131 + 1);
        sub_C7D6A0(v128, 24LL * v129, 8);
        v7 = 24LL * v127;
        sub_C7D6A0(v126[6], v7, 8);
        if ( v125 != v126 )
        {
          v7 = v126[0] + 1LL;
          j_j___libc_free_0(v125, v126[0] + 1LL);
        }
        v40 = 32LL * v123;
        v41 = &v122[v40];
        if ( v122 == &v122[v40] )
        {
LABEL_80:
          if ( v41 != &v124 )
            _libc_free(v41, v7);
          v50 = v120;
          v51 = v119;
          if ( v120 == v119 )
          {
LABEL_97:
            if ( v51 )
            {
              v7 = v121 - v51;
              j_j___libc_free_0(v51, v121 - v51);
            }
            v57 = (_QWORD *)v116;
            if ( v117 == (_QWORD *)v116 )
              goto LABEL_129;
            v91 = v11;
            v58 = v117;
            while ( 1 )
            {
              v59 = v57[9];
              v60 = v57[8];
              if ( v59 != v60 )
              {
                do
                {
                  v61 = *(_QWORD *)(v60 + 8);
                  if ( v61 != v60 + 24 )
                  {
                    v7 = *(_QWORD *)(v60 + 24) + 1LL;
                    j_j___libc_free_0(v61, v7);
                  }
                  v60 += 40;
                }
                while ( v59 != v60 );
                v60 = v57[8];
              }
              if ( v60 )
              {
                v7 = v57[10] - v60;
                j_j___libc_free_0(v60, v7);
              }
              v62 = (_QWORD *)v57[4];
              if ( v62 != v57 + 6 )
              {
                v7 = v57[6] + 1LL;
                j_j___libc_free_0(v62, v7);
              }
              v63 = v57[2];
              v64 = v57[1];
              if ( v63 != v64 )
                break;
LABEL_125:
              if ( v64 )
              {
                v7 = v57[3] - v64;
                j_j___libc_free_0(v64, v7);
              }
              v57 += 11;
              if ( v58 == v57 )
              {
                v11 = v91;
                v57 = (_QWORD *)v116;
LABEL_129:
                if ( v57 )
                {
                  v7 = v118 - (_QWORD)v57;
                  j_j___libc_free_0(v57, v118 - (_QWORD)v57);
                }
                if ( v11 )
                {
                  sub_9CD560(v11);
                  v7 = 584;
                  j_j___libc_free_0(v11, 584);
                }
                goto LABEL_134;
              }
            }
            while ( 1 )
            {
              v65 = *(volatile signed __int32 **)(v64 + 8);
              if ( !v65 )
                goto LABEL_112;
              if ( &_pthread_key_create )
              {
                v66 = _InterlockedExchangeAdd(v65 + 2, 0xFFFFFFFF);
              }
              else
              {
                v66 = *((_DWORD *)v65 + 2);
                v7 = (unsigned int)(v66 - 1);
                *((_DWORD *)v65 + 2) = v7;
              }
              if ( v66 != 1 )
                goto LABEL_112;
              v67 = *(void (**)())(*(_QWORD *)v65 + 16LL);
              if ( v67 != nullsub_25 )
                ((void (__fastcall *)(volatile signed __int32 *))v67)(v65);
              if ( &_pthread_key_create )
              {
                v68 = _InterlockedExchangeAdd(v65 + 3, 0xFFFFFFFF);
              }
              else
              {
                v68 = *((_DWORD *)v65 + 3);
                *((_DWORD *)v65 + 3) = v68 - 1;
              }
              if ( v68 != 1 )
                goto LABEL_112;
              v69 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v65 + 24LL);
              if ( v69 == sub_9C26E0 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v65 + 8LL))(v65);
                v64 += 16;
                if ( v63 == v64 )
                {
LABEL_124:
                  v64 = v57[1];
                  goto LABEL_125;
                }
              }
              else
              {
                v69((__int64)v65);
LABEL_112:
                v64 += 16;
                if ( v63 == v64 )
                  goto LABEL_124;
              }
            }
          }
          while ( 1 )
          {
            v52 = *(volatile signed __int32 **)(v51 + 8);
            if ( !v52 )
              goto LABEL_84;
            if ( &_pthread_key_create )
            {
              v53 = _InterlockedExchangeAdd(v52 + 2, 0xFFFFFFFF);
            }
            else
            {
              v53 = *((_DWORD *)v52 + 2);
              *((_DWORD *)v52 + 2) = v53 - 1;
            }
            if ( v53 != 1 )
              goto LABEL_84;
            v54 = *(void (**)())(*(_QWORD *)v52 + 16LL);
            if ( v54 != nullsub_25 )
              ((void (__fastcall *)(volatile signed __int32 *))v54)(v52);
            if ( &_pthread_key_create )
            {
              v55 = _InterlockedExchangeAdd(v52 + 3, 0xFFFFFFFF);
            }
            else
            {
              v55 = *((_DWORD *)v52 + 3);
              *((_DWORD *)v52 + 3) = v55 - 1;
            }
            if ( v55 != 1 )
              goto LABEL_84;
            v56 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v52 + 24LL);
            if ( v56 == sub_9C26E0 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v52 + 8LL))(v52);
              v51 += 16;
              if ( v50 == v51 )
              {
LABEL_96:
                v51 = v119;
                goto LABEL_97;
              }
            }
            else
            {
              v56((__int64)v52);
LABEL_84:
              v51 += 16;
              if ( v50 == v51 )
                goto LABEL_96;
            }
          }
        }
        v93 = v122;
        v90 = v11;
        v42 = &v122[v40];
        while ( 1 )
        {
          v43 = *((_QWORD *)v42 - 3);
          v44 = *((_QWORD *)v42 - 2);
          v42 -= 32;
          v45 = v43;
          if ( v44 != v43 )
            break;
LABEL_76:
          if ( v43 )
          {
            v7 = *((_QWORD *)v42 + 3) - v43;
            j_j___libc_free_0(v43, v7);
          }
          if ( v93 == v42 )
          {
            v11 = v90;
            v41 = v122;
            goto LABEL_80;
          }
        }
        while ( 1 )
        {
          v46 = *(volatile signed __int32 **)(v45 + 8);
          if ( !v46 )
            goto LABEL_63;
          if ( &_pthread_key_create )
          {
            v47 = _InterlockedExchangeAdd(v46 + 2, 0xFFFFFFFF);
          }
          else
          {
            v47 = *((_DWORD *)v46 + 2);
            v7 = (unsigned int)(v47 - 1);
            *((_DWORD *)v46 + 2) = v7;
          }
          if ( v47 != 1 )
            goto LABEL_63;
          v48 = *(void (**)())(*(_QWORD *)v46 + 16LL);
          if ( v48 != nullsub_25 )
            ((void (__fastcall *)(volatile signed __int32 *))v48)(v46);
          if ( &_pthread_key_create )
          {
            v49 = _InterlockedExchangeAdd(v46 + 3, 0xFFFFFFFF);
          }
          else
          {
            v49 = *((_DWORD *)v46 + 3);
            v7 = (unsigned int)(v49 - 1);
            *((_DWORD *)v46 + 3) = v7;
          }
          if ( v49 != 1 )
            goto LABEL_63;
          v7 = *(_QWORD *)(*(_QWORD *)v46 + 24LL);
          if ( (__int64 (__fastcall *)(__int64))v7 == sub_9C26E0 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v46 + 8LL))(v46);
            v45 += 16;
            if ( v44 == v45 )
            {
LABEL_75:
              v43 = *((_QWORD *)v42 + 1);
              goto LABEL_76;
            }
          }
          else
          {
            ((void (__fastcall *)(volatile signed __int32 *))v7)(v46);
LABEL_63:
            v45 += 16;
            if ( v44 == v45 )
              goto LABEL_75;
          }
        }
      }
    }
    j_j___libc_free_0(v39, v137 - v39);
    goto LABEL_51;
  }
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v9;
LABEL_134:
  v70 = 32LL * (unsigned int)v104;
  v71 = &v103[v70];
  if ( v103 == &v103[v70] )
    goto LABEL_155;
  v94 = v103;
  v72 = &v103[v70];
  do
  {
    v73 = *((_QWORD *)v72 - 3);
    v74 = *((_QWORD *)v72 - 2);
    v72 -= 32;
    v75 = v73;
    if ( v74 != v73 )
    {
      while ( 1 )
      {
        v76 = *(volatile signed __int32 **)(v75 + 8);
        if ( !v76 )
          goto LABEL_138;
        if ( &_pthread_key_create )
        {
          v77 = _InterlockedExchangeAdd(v76 + 2, 0xFFFFFFFF);
        }
        else
        {
          v77 = *((_DWORD *)v76 + 2);
          *((_DWORD *)v76 + 2) = v77 - 1;
        }
        if ( v77 != 1 )
          goto LABEL_138;
        v78 = *(void (**)())(*(_QWORD *)v76 + 16LL);
        if ( v78 != nullsub_25 )
          ((void (__fastcall *)(volatile signed __int32 *))v78)(v76);
        if ( &_pthread_key_create )
        {
          v79 = _InterlockedExchangeAdd(v76 + 3, 0xFFFFFFFF);
        }
        else
        {
          v79 = *((_DWORD *)v76 + 3);
          *((_DWORD *)v76 + 3) = v79 - 1;
        }
        if ( v79 != 1 )
          goto LABEL_138;
        v80 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v76 + 24LL);
        if ( v80 == sub_9C26E0 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v76 + 8LL))(v76);
          v75 += 16;
          if ( v74 == v75 )
          {
LABEL_150:
            v73 = *((_QWORD *)v72 + 1);
            break;
          }
        }
        else
        {
          v80((__int64)v76);
LABEL_138:
          v75 += 16;
          if ( v74 == v75 )
            goto LABEL_150;
        }
      }
    }
    if ( v73 )
    {
      v7 = *((_QWORD *)v72 + 3) - v73;
      j_j___libc_free_0(v73, v7);
    }
  }
  while ( v94 != v72 );
  v71 = v103;
LABEL_155:
  if ( v71 != v105 )
    _libc_free(v71, v7);
  v81 = v101;
  v82 = v100;
  if ( v101 != v100 )
  {
    while ( 1 )
    {
      v83 = *(volatile signed __int32 **)(v82 + 8);
      if ( !v83 )
        goto LABEL_159;
      if ( &_pthread_key_create )
      {
        v84 = _InterlockedExchangeAdd(v83 + 2, 0xFFFFFFFF);
      }
      else
      {
        v84 = *((_DWORD *)v83 + 2);
        *((_DWORD *)v83 + 2) = v84 - 1;
      }
      if ( v84 != 1 )
        goto LABEL_159;
      v85 = *(void (**)())(*(_QWORD *)v83 + 16LL);
      if ( v85 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v85)(v83);
      if ( &_pthread_key_create )
      {
        v86 = _InterlockedExchangeAdd(v83 + 3, 0xFFFFFFFF);
      }
      else
      {
        v86 = *((_DWORD *)v83 + 3);
        *((_DWORD *)v83 + 3) = v86 - 1;
      }
      if ( v86 != 1 )
        goto LABEL_159;
      v87 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v83 + 24LL);
      if ( v87 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v83 + 8LL))(v83);
        v82 += 16;
        if ( v81 == v82 )
        {
LABEL_171:
          v82 = v100;
          break;
        }
      }
      else
      {
        v87((__int64)v83);
LABEL_159:
        v82 += 16;
        if ( v81 == v82 )
          goto LABEL_171;
      }
    }
  }
  if ( v82 )
    j_j___libc_free_0(v82, v102 - v82);
  return a1;
}
