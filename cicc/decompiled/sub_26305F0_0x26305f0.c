// Function: sub_26305F0
// Address: 0x26305f0
//
void __fastcall sub_26305F0(__int64 a1, _BYTE *a2, __int64 a3, _QWORD *a4)
{
  __int64 *v7; // rsi
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // r14
  char v15; // r15
  unsigned __int64 *v16; // rbx
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  unsigned __int64 *v19; // r14
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  int v27; // r15d
  unsigned int v28; // eax
  int v29; // eax
  const void **v30; // rcx
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 *v36; // r12
  unsigned __int64 *v37; // r15
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r13
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r12
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  __int64 v47; // r15
  unsigned __int64 v48; // r12
  __int64 v49; // rbx
  unsigned __int64 v50; // r13
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  __int64 v55; // rbx
  unsigned __int64 v56; // r12
  unsigned __int64 v57; // rdi
  __int64 v58; // rbx
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rdi
  char *v61; // rsi
  void (__fastcall *v62)(__int64, __int64 **); // rax
  __int64 v63; // rax
  _BYTE *v64; // rbx
  char v65; // al
  _QWORD *v66; // rax
  unsigned __int64 v67; // rcx
  __int64 v68; // r8
  _QWORD *v69; // rdx
  __int64 v70; // rdi
  __int64 v71; // rsi
  unsigned __int64 *v72; // rdx
  __int64 v73; // rdi
  __int64 v74; // rsi
  char *v75; // rsi
  __int64 v76; // [rsp+8h] [rbp-208h]
  __int64 v77; // [rsp+10h] [rbp-200h]
  __int64 *v78; // [rsp+20h] [rbp-1F0h]
  int v79; // [rsp+34h] [rbp-1DCh]
  char v80; // [rsp+38h] [rbp-1D8h]
  char v81; // [rsp+3Ch] [rbp-1D4h]
  char v82; // [rsp+40h] [rbp-1D0h]
  char v83; // [rsp+44h] [rbp-1CCh]
  char v84; // [rsp+48h] [rbp-1C8h]
  _QWORD *v85; // [rsp+48h] [rbp-1C8h]
  _QWORD *v86; // [rsp+48h] [rbp-1C8h]
  char v87; // [rsp+58h] [rbp-1B8h]
  __int64 *v88; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 *v89; // [rsp+60h] [rbp-1B0h]
  __int64 *v90; // [rsp+88h] [rbp-188h] BYREF
  unsigned __int64 *v91; // [rsp+90h] [rbp-180h] BYREF
  __int64 v92; // [rsp+98h] [rbp-178h]
  unsigned __int64 v93[2]; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v94; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v95; // [rsp+B8h] [rbp-158h]
  __int64 v96; // [rsp+C0h] [rbp-150h]
  unsigned __int64 v97[4]; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v98[4]; // [rsp+F0h] [rbp-120h] BYREF
  __int64 v99[4]; // [rsp+110h] [rbp-100h] BYREF
  unsigned __int64 v100; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v101; // [rsp+138h] [rbp-D8h]
  __int64 v102; // [rsp+140h] [rbp-D0h]
  unsigned __int64 v103; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v104; // [rsp+158h] [rbp-B8h]
  __int64 v105; // [rsp+160h] [rbp-B0h]
  unsigned __int64 v106; // [rsp+170h] [rbp-A0h] BYREF
  __int64 v107; // [rsp+178h] [rbp-98h]
  __int64 v108; // [rsp+180h] [rbp-90h]
  unsigned __int64 *v109; // [rsp+190h] [rbp-80h] BYREF
  unsigned __int64 *v110; // [rsp+198h] [rbp-78h]
  __int64 v111; // [rsp+1A0h] [rbp-70h]
  __int64 *v112; // [rsp+1B0h] [rbp-60h] BYREF
  char *v113; // [rsp+1B8h] [rbp-58h]
  _QWORD v114[2]; // [rsp+1C0h] [rbp-50h] BYREF
  char v115; // [rsp+1D0h] [rbp-40h]
  char v116; // [rsp+1D1h] [rbp-3Fh]

  v94 = 0;
  v95 = 0;
  v96 = 0;
  if ( a2 )
  {
    v112 = v114;
    sub_2619AF0((__int64 *)&v112, a2, (__int64)&a2[a3]);
    v7 = v112;
  }
  else
  {
    v113 = 0;
    v112 = v114;
    v7 = v114;
    LOBYTE(v114[0]) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64, _QWORD, unsigned __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v7,
         1,
         0,
         &v106) )
  {
    sub_262F0A0(a1, &v94);
    (*(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v109);
  }
  if ( v112 != v114 )
    j_j___libc_free_0((unsigned __int64)v112);
  if ( sub_C93C90((__int64)a2, a3, 0, (unsigned __int64 *)&v112) )
  {
    v62 = *(void (__fastcall **)(__int64, __int64 **))(*(_QWORD *)a1 + 248LL);
    v112 = (__int64 *)"key not an integer";
    v116 = 1;
    v115 = 3;
    v62(a1, &v112);
    sub_2627CA0((__int64)&v94);
    return;
  }
  v9 = (__int64 *)a4[2];
  v10 = a4 + 1;
  LOBYTE(v106) = 0;
  v89 = a4 + 1;
  v90 = v112;
  if ( v9 )
  {
    do
    {
      while ( 1 )
      {
        v11 = v9[2];
        v12 = v9[3];
        if ( (unsigned __int64)v112 <= v9[4] )
          break;
        v9 = (__int64 *)v9[3];
        if ( !v12 )
          goto LABEL_13;
      }
      v10 = v9;
      v9 = (__int64 *)v9[2];
    }
    while ( v11 );
LABEL_13:
    v78 = v10;
    if ( v89 != (unsigned __int64 *)v10 && (unsigned __int64)v112 >= v10[4] )
      goto LABEL_16;
  }
  else
  {
    v78 = a4 + 1;
  }
  v112 = (__int64 *)&v106;
  v109 = (unsigned __int64 *)&v90;
  v78 = (__int64 *)sub_262CD10(a4, (__int64)v78, &v109, (_BYTE **)&v112);
LABEL_16:
  v13 = a4;
  v77 = v95;
  v14 = v94 + 16;
  if ( v94 != v95 )
  {
    while ( 1 )
    {
      v87 = *(_BYTE *)(v14 - 5);
      v15 = *(_BYTE *)(v14 - 12) & 3;
      v84 = *(_BYTE *)(v14 - 6);
      v83 = *(_BYTE *)(v14 - 7);
      v82 = *(_BYTE *)(v14 - 8);
      v80 = *(_BYTE *)(v14 - 16) & 0xF;
      v81 = *(_BYTE *)(v14 - 4) & 1;
      if ( !*(_BYTE *)(v14 + 8) )
      {
        v92 = 0;
        v91 = v93;
        v16 = *(unsigned __int64 **)(v14 + 24);
        v17 = *(_QWORD *)(v14 + 16);
        v18 = ((__int64)v16 - v17) >> 3;
        if ( v18 )
        {
          sub_C8D5F0((__int64)&v91, v93, v18, 8u, v17, v8);
          v16 = *(unsigned __int64 **)(v14 + 24);
          v17 = *(_QWORD *)(v14 + 16);
        }
        if ( v16 != (unsigned __int64 *)v17 )
        {
          v76 = v14;
          v19 = (unsigned __int64 *)v17;
          while ( 1 )
          {
            v20 = (unsigned __int64 *)v13[2];
            LOBYTE(v106) = 0;
            if ( v20 )
            {
              v21 = *v19;
              v22 = v89;
              do
              {
                while ( 1 )
                {
                  v23 = v20[2];
                  v24 = v20[3];
                  if ( v20[4] >= v21 )
                    break;
                  v20 = (unsigned __int64 *)v20[3];
                  if ( !v24 )
                    goto LABEL_27;
                }
                v22 = v20;
                v20 = (unsigned __int64 *)v20[2];
              }
              while ( v23 );
LABEL_27:
              if ( v89 != v22 && v21 >= v22[4] )
                goto LABEL_30;
            }
            else
            {
              v22 = v89;
            }
            v109 = v19;
            v112 = (__int64 *)&v106;
            v22 = sub_262CD10(v13, (__int64)v22, &v109, (_BYTE **)&v112);
LABEL_30:
            v25 = (unsigned int)v92;
            v26 = (unsigned __int64)(v22 + 4) & 0xFFFFFFFFFFFFFFF8LL;
            if ( (unsigned __int64)(unsigned int)v92 + 1 > HIDWORD(v92) )
            {
              sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 8u, v17, v8);
              v25 = (unsigned int)v92;
              v26 = (unsigned __int64)(v22 + 4) & 0xFFFFFFFFFFFFFFF8LL;
            }
            ++v19;
            v91[v25] = v26;
            LODWORD(v92) = v92 + 1;
            if ( v16 == v19 )
            {
              v14 = v76;
              break;
            }
          }
        }
        v27 = 16 * (v15 & 3);
        v93[1] = 0;
        v93[0] = (unsigned __int64)&v94;
        v28 = v27 | v80 & 0xF | v79 & 0xFFFFFFC0;
        LOBYTE(v28) = v27 & 0x3F | v80 & 0xF;
        v29 = ((v83 & 1) << 7) | ((v82 & 1) << 6) | v28;
        BYTE1(v29) &= 0xFCu;
        v30 = (const void **)((unsigned __int8)(v84 & 1) << 8);
        v31 = ((v87 & 1) << 9) | (unsigned int)v30 | v29;
        BYTE1(v31) &= ~4u;
        v79 = ((v81 & 1) << 10) | v31;
        v97[0] = *(_QWORD *)(v14 + 40);
        v97[1] = *(_QWORD *)(v14 + 48);
        v97[2] = *(_QWORD *)(v14 + 56);
        v32 = *(_QWORD *)(v14 + 64);
        *(_QWORD *)(v14 + 56) = 0;
        *(_QWORD *)(v14 + 48) = 0;
        *(_QWORD *)(v14 + 40) = 0;
        v98[0] = v32;
        v98[1] = *(_QWORD *)(v14 + 72);
        v98[2] = *(_QWORD *)(v14 + 80);
        v33 = *(_QWORD *)(v14 + 88);
        *(_QWORD *)(v14 + 80) = 0;
        *(_QWORD *)(v14 + 72) = 0;
        *(_QWORD *)(v14 + 64) = 0;
        v99[0] = v33;
        v99[1] = *(_QWORD *)(v14 + 96);
        v99[2] = *(_QWORD *)(v14 + 104);
        v34 = *(_QWORD *)(v14 + 112);
        *(_QWORD *)(v14 + 104) = 0;
        *(_QWORD *)(v14 + 96) = 0;
        *(_QWORD *)(v14 + 88) = 0;
        v100 = v34;
        v101 = *(_QWORD *)(v14 + 120);
        v102 = *(_QWORD *)(v14 + 128);
        v35 = *(_QWORD *)(v14 + 136);
        *(_QWORD *)(v14 + 128) = 0;
        *(_QWORD *)(v14 + 120) = 0;
        *(_QWORD *)(v14 + 112) = 0;
        v103 = v35;
        v104 = *(_QWORD *)(v14 + 144);
        v105 = *(_QWORD *)(v14 + 152);
        *(_QWORD *)(v14 + 152) = 0;
        *(_QWORD *)(v14 + 144) = 0;
        *(_QWORD *)(v14 + 136) = 0;
        v106 = 0;
        v107 = 0;
        v108 = 0;
        v109 = 0;
        v110 = 0;
        v111 = 0;
        v112 = 0;
        v113 = 0;
        v114[0] = 0;
        v113 = sub_9EB710(0, 0, 0, v30);
        v88 = (__int64 *)sub_22077B0(0x70u);
        if ( v88 )
          sub_9C6E00(
            (__int64)v88,
            v79,
            0,
            0,
            (__int64)&v91,
            (__int64)v93,
            v97,
            v98,
            v99,
            (__int64 *)&v100,
            (__int64 *)&v103,
            (__int64 *)&v106,
            (__int64 *)&v109,
            (__int64)&v112);
        v36 = (unsigned __int64 *)v112;
        if ( v113 != (char *)v112 )
        {
          v85 = v13;
          v37 = (unsigned __int64 *)v113;
          do
          {
            v38 = (unsigned __int64 *)v36[12];
            v39 = (unsigned __int64 *)v36[11];
            if ( v38 != v39 )
            {
              do
              {
                if ( *v39 )
                  j_j___libc_free_0(*v39);
                v39 += 3;
              }
              while ( v38 != v39 );
              v39 = (unsigned __int64 *)v36[11];
            }
            if ( v39 )
              j_j___libc_free_0((unsigned __int64)v39);
            v40 = v36[9];
            v41 = v36[8];
            if ( v40 != v41 )
            {
              do
              {
                v42 = *(_QWORD *)(v41 + 8);
                if ( v42 != v41 + 24 )
                  _libc_free(v42);
                v41 += 72LL;
              }
              while ( v40 != v41 );
              v41 = v36[8];
            }
            if ( v41 )
              j_j___libc_free_0(v41);
            if ( (unsigned __int64 *)*v36 != v36 + 3 )
              _libc_free(*v36);
            v36 += 14;
          }
          while ( v37 != v36 );
          v13 = v85;
          v36 = (unsigned __int64 *)v112;
        }
        if ( v36 )
          j_j___libc_free_0((unsigned __int64)v36);
        v43 = v110;
        v44 = v109;
        if ( v110 != v109 )
        {
          do
          {
            v45 = v44[9];
            if ( (unsigned __int64 *)v45 != v44 + 11 )
              _libc_free(v45);
            v46 = v44[1];
            if ( (unsigned __int64 *)v46 != v44 + 3 )
              _libc_free(v46);
            v44 += 17;
          }
          while ( v43 != v44 );
          v44 = v109;
        }
        if ( v44 )
          j_j___libc_free_0((unsigned __int64)v44);
        v47 = v107;
        v48 = v106;
        if ( v107 != v106 )
        {
          v86 = v13;
          do
          {
            v49 = *(_QWORD *)(v48 + 48);
            v50 = *(_QWORD *)(v48 + 40);
            if ( v49 != v50 )
            {
              do
              {
                if ( *(_DWORD *)(v50 + 40) > 0x40u )
                {
                  v51 = *(_QWORD *)(v50 + 32);
                  if ( v51 )
                    j_j___libc_free_0_0(v51);
                }
                if ( *(_DWORD *)(v50 + 24) > 0x40u )
                {
                  v52 = *(_QWORD *)(v50 + 16);
                  if ( v52 )
                    j_j___libc_free_0_0(v52);
                }
                v50 += 48LL;
              }
              while ( v49 != v50 );
              v50 = *(_QWORD *)(v48 + 40);
            }
            if ( v50 )
              j_j___libc_free_0(v50);
            if ( *(_DWORD *)(v48 + 32) > 0x40u )
            {
              v53 = *(_QWORD *)(v48 + 24);
              if ( v53 )
                j_j___libc_free_0_0(v53);
            }
            if ( *(_DWORD *)(v48 + 16) > 0x40u )
            {
              v54 = *(_QWORD *)(v48 + 8);
              if ( v54 )
                j_j___libc_free_0_0(v54);
            }
            v48 += 64LL;
          }
          while ( v47 != v48 );
          v13 = v86;
          v48 = v106;
        }
        if ( v48 )
          j_j___libc_free_0(v48);
        v55 = v104;
        v56 = v103;
        if ( v104 != v103 )
        {
          do
          {
            v57 = *(_QWORD *)(v56 + 16);
            if ( v57 )
              j_j___libc_free_0(v57);
            v56 += 40LL;
          }
          while ( v55 != v56 );
          v56 = v103;
        }
        if ( v56 )
          j_j___libc_free_0(v56);
        v58 = v101;
        v59 = v100;
        if ( v101 != v100 )
        {
          do
          {
            v60 = *(_QWORD *)(v59 + 16);
            if ( v60 )
              j_j___libc_free_0(v60);
            v59 += 40LL;
          }
          while ( v58 != v59 );
          v59 = v100;
        }
        if ( v59 )
          j_j___libc_free_0(v59);
        if ( v99[0] )
          j_j___libc_free_0(v99[0]);
        if ( v98[0] )
          j_j___libc_free_0(v98[0]);
        if ( v97[0] )
          j_j___libc_free_0(v97[0]);
        v112 = v88;
        v61 = (char *)v78[8];
        if ( v61 == (char *)v78[9] )
        {
          sub_9D0210(v78 + 7, v61, &v112);
          v88 = v112;
        }
        else
        {
          if ( v61 )
          {
            *(_QWORD *)v61 = v88;
            v78[8] += 8;
LABEL_113:
            if ( (__int64 *)v93[0] != &v94 )
              _libc_free(v93[0]);
            if ( v91 != v93 )
              _libc_free((unsigned __int64)v91);
            goto LABEL_117;
          }
          v78[8] = 8;
        }
        if ( v88 )
          (*(void (__fastcall **)(__int64 *))(*v88 + 8))(v88);
        goto LABEL_113;
      }
      v63 = sub_22077B0(0x48u);
      v64 = (_BYTE *)v63;
      if ( v63 )
      {
        *(_DWORD *)(v63 + 8) = 0;
        *(_QWORD *)(v63 + 16) = 0;
        *(_QWORD *)(v63 + 24) = 0;
        *(_QWORD *)(v63 + 32) = 0;
        v65 = *(_BYTE *)(v63 + 13);
        *((_QWORD *)v64 + 6) = 0;
        *((_QWORD *)v64 + 7) = 0;
        v64[12] = (v83 << 7) | ((v82 & 1) << 6) | (16 * v15) | v80;
        *((_QWORD *)v64 + 8) = 0;
        v64[13] = (4 * v81) | (2 * (v87 & 1)) & 0xFB | v84 & 1 | v65 & 0xF8;
        *((_QWORD *)v64 + 5) = v64 + 56;
        *(_QWORD *)v64 = &unk_49D9790;
      }
      v66 = (_QWORD *)v13[2];
      LOBYTE(v106) = 0;
      if ( v66 )
      {
        v67 = *(_QWORD *)v14;
        v68 = (__int64)v89;
        v69 = v66;
        do
        {
          while ( 1 )
          {
            v70 = v69[2];
            v71 = v69[3];
            if ( v69[4] >= v67 )
              break;
            v69 = (_QWORD *)v69[3];
            if ( !v71 )
              goto LABEL_129;
          }
          v68 = (__int64)v69;
          v69 = (_QWORD *)v69[2];
        }
        while ( v70 );
LABEL_129:
        if ( v89 != (unsigned __int64 *)v68 && v67 >= *(_QWORD *)(v68 + 32) )
          goto LABEL_133;
      }
      else
      {
        v68 = (__int64)v89;
      }
      v109 = (unsigned __int64 *)v14;
      v112 = (__int64 *)&v106;
      sub_262CD10(v13, v68, &v109, (_BYTE **)&v112);
      v66 = (_QWORD *)v13[2];
      if ( v66 )
      {
        v67 = *(_QWORD *)v14;
LABEL_133:
        v72 = v89;
        do
        {
          while ( 1 )
          {
            v73 = v66[2];
            v74 = v66[3];
            if ( v66[4] >= v67 )
              break;
            v66 = (_QWORD *)v66[3];
            if ( !v74 )
              goto LABEL_137;
          }
          v72 = v66;
          v66 = (_QWORD *)v66[2];
        }
        while ( v73 );
LABEL_137:
        if ( v89 != v72 && v67 < v72[4] )
          v72 = v89;
        goto LABEL_140;
      }
      v72 = v89;
LABEL_140:
      *((_QWORD *)v64 + 8) = 0;
      v112 = (__int64 *)v64;
      *((_QWORD *)v64 + 7) = (unsigned __int64)(v72 + 4) & 0xFFFFFFFFFFFFFFF8LL;
      v75 = (char *)v78[8];
      if ( v75 != (char *)v78[9] )
      {
        if ( v75 )
        {
          *(_QWORD *)v75 = v64;
          v78[8] += 8;
          goto LABEL_117;
        }
        v78[8] = 8;
LABEL_149:
        (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v64 + 8LL))(v64);
        goto LABEL_117;
      }
      sub_9D0210(v78 + 7, v75, &v112);
      v64 = v112;
      if ( v112 )
        goto LABEL_149;
LABEL_117:
      if ( v77 == v14 + 160 )
        break;
      v14 += 176;
    }
  }
  sub_2627CA0((__int64)&v94);
}
