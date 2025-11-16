// Function: sub_24900F0
// Address: 0x24900f0
//
__int64 __fastcall sub_24900F0(__int64 a1, unsigned __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r15
  const char *v5; // rax
  __int64 v7; // rbx
  __int64 *v8; // rdx
  bool v9; // zf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 *v12; // rsi
  char v13; // al
  _QWORD *v14; // r9
  __int64 *v15; // rax
  __int64 *v16; // rcx
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64); // rax
  char v21; // al
  unsigned __int64 v22; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  _QWORD *v30; // r12
  __int64 **v31; // rax
  _QWORD *v32; // rbx
  __int64 **v33; // r12
  _QWORD *v34; // r15
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rax
  __m128i *v38; // rdx
  __int64 v39; // rdi
  __m128i si128; // xmm0
  __int64 v41; // rax
  __m128i *v42; // rdx
  __int64 v43; // rdi
  __m128i v44; // xmm0
  char *v45; // r13
  __int64 v46; // r15
  size_t v47; // rdx
  __int64 v48; // rax
  size_t v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // r13
  _QWORD *v53; // r8
  unsigned int v54; // eax
  __int64 v55; // r13
  __int64 v56; // r12
  _QWORD *v57; // rbx
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  unsigned int v60; // ecx
  __int64 v61; // rsi
  __int64 *v62; // r13
  _QWORD *v63; // rax
  __m128i *v64; // rdx
  __m128i v65; // xmm0
  unsigned __int64 *v66; // r12
  unsigned __int64 *i; // rbx
  unsigned __int64 v68; // r15
  __int64 v69; // rdi
  _BYTE *v70; // rax
  _QWORD *v71; // rdi
  _BYTE *v72; // rax
  unsigned int v73; // eax
  __int64 v74; // rbx
  __int64 v75; // r12
  _QWORD *v76; // r13
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rdi
  __int64 v79; // [rsp+18h] [rbp-148h]
  __int64 v80; // [rsp+20h] [rbp-140h]
  _QWORD **v81; // [rsp+30h] [rbp-130h]
  __int64 v83; // [rsp+38h] [rbp-128h]
  __int64 v84; // [rsp+38h] [rbp-128h]
  const char *v85; // [rsp+40h] [rbp-120h]
  __int64 *v86; // [rsp+40h] [rbp-120h]
  __int64 v87; // [rsp+40h] [rbp-120h]
  _QWORD *v88; // [rsp+48h] [rbp-118h]
  _QWORD *v89; // [rsp+48h] [rbp-118h]
  __int64 v90; // [rsp+48h] [rbp-118h]
  __int64 v91; // [rsp+58h] [rbp-108h] BYREF
  __int64 v92; // [rsp+60h] [rbp-100h] BYREF
  __int64 v93; // [rsp+68h] [rbp-F8h] BYREF
  unsigned __int64 v94; // [rsp+70h] [rbp-F0h]
  __int64 v95; // [rsp+78h] [rbp-E8h] BYREF
  _QWORD **v96; // [rsp+80h] [rbp-E0h] BYREF
  char v97; // [rsp+88h] [rbp-D8h]
  __int64 v98; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v99; // [rsp+98h] [rbp-C8h]
  _QWORD *v100; // [rsp+A0h] [rbp-C0h]
  unsigned int v101; // [rsp+A8h] [rbp-B8h]
  unsigned __int8 *v102[2]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v103; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned __int64 v104; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-88h] BYREF
  unsigned __int64 v106; // [rsp+E0h] [rbp-80h]
  __int64 *v107; // [rsp+E8h] [rbp-78h]
  __int64 *v108; // [rsp+F0h] [rbp-70h]
  __int64 v109; // [rsp+F8h] [rbp-68h]
  const char *v110; // [rsp+100h] [rbp-60h] BYREF
  __int64 v111; // [rsp+108h] [rbp-58h] BYREF
  _QWORD *v112; // [rsp+110h] [rbp-50h]
  __int64 *v113; // [rsp+118h] [rbp-48h]
  __int64 *v114; // [rsp+120h] [rbp-40h]
  __int64 v115; // [rsp+128h] [rbp-38h]

  v4 = a1;
  v88 = a3 + 3;
  if ( a3 + 3 == (_QWORD *)(a3[3] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return v4;
  }
  v5 = (const char *)*a3;
  v7 = (__int64)a3;
  v8 = (__int64 *)a2[4];
  LOWORD(v114) = 257;
  v85 = v5;
  LOWORD(v108) = 260;
  v104 = (unsigned __int64)a2;
  sub_EDC610((__int64)&v96, (void **)&v104, v8, (void **)&v110);
  v9 = (v97 & 1) == 0;
  v10 = (unsigned __int64)v96;
  v97 &= ~2u;
  if ( !v9 )
  {
    v96 = 0;
    v11 = v10 & 0xFFFFFFFFFFFFFFFELL;
    if ( v11 )
    {
      v111 = (__int64)a2;
      v89 = (_QWORD *)v11;
      v12 = (__int64 *)&unk_4F84052;
      v110 = v85;
      v91 = 0;
      v92 = 0;
      v93 = 0;
      v13 = (*(__int64 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v11 + 48LL))(v11, &unk_4F84052);
      v14 = v89;
      if ( v13 )
      {
        v15 = (__int64 *)v89[2];
        v16 = (__int64 *)v89[1];
        v94 = 1;
        v86 = v15;
        if ( v16 == v15 )
        {
          v18 = 1;
        }
        else
        {
          v17 = v16;
          do
          {
            v98 = *v17;
            *v17 = 0;
            sub_2486CB0((__int64 *)v102, &v98, (__int64)&v110);
            v12 = &v95;
            v95 = v94 | 1;
            sub_9CDB40(&v104, (unsigned __int64 *)&v95, (unsigned __int64 *)v102);
            v94 = v104 | 1;
            if ( (v95 & 1) != 0 || (v95 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v95, (__int64)&v95);
            if ( ((__int64)v102[0] & 1) != 0 || ((unsigned __int64)v102[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(v102, (__int64)&v95);
            if ( v98 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v98 + 8LL))(v98);
            ++v17;
          }
          while ( v86 != v17 );
          v14 = v89;
          v4 = a1;
          v18 = v94 | 1;
        }
        v102[0] = (unsigned __int8 *)v18;
        (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
      }
      else
      {
        v12 = (__int64 *)&v104;
        v104 = (unsigned __int64)v89;
        sub_2486CB0((__int64 *)v102, &v104, (__int64)&v110);
        if ( v104 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v104 + 8LL))(v104);
      }
      if ( ((unsigned __int64)v102[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      if ( (v93 & 1) != 0 || (v93 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v93, (__int64)v12);
      if ( (v92 & 1) != 0 || (v92 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v92, (__int64)v12);
      *(_QWORD *)(v4 + 48) = 0;
      *(_QWORD *)(v4 + 8) = v4 + 32;
      *(_QWORD *)(v4 + 56) = v4 + 80;
      *(_QWORD *)(v4 + 16) = 0x100000002LL;
      *(_QWORD *)(v4 + 64) = 2;
      *(_QWORD *)(v4 + 32) = &qword_4F82400;
      v19 = v91;
      *(_DWORD *)(v4 + 72) = 0;
      *(_BYTE *)(v4 + 76) = 1;
      *(_DWORD *)(v4 + 24) = 0;
      *(_BYTE *)(v4 + 28) = 1;
      *(_QWORD *)v4 = 1;
      if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v91, (__int64)v12);
      goto LABEL_30;
    }
    v80 = a1 + 32;
    v79 = a1 + 80;
    goto LABEL_36;
  }
  v81 = v96;
  v80 = a1 + 32;
  v96 = 0;
  v79 = a1 + 80;
  if ( !v10 )
  {
LABEL_36:
    v12 = (__int64 *)&v104;
    v110 = "Cannot get MemProfReader";
    v24 = *a2;
    LOWORD(v114) = 261;
    v111 = 24;
    v106 = v24;
    v105 = 23;
    v104 = (unsigned __int64)&unk_49D9CA8;
    v107 = (__int64 *)&v110;
    sub_B6EB20((__int64)v85, (__int64)&v104);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 8) = v80;
    *(_DWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 56) = v79;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_30;
  }
  v20 = (__int64 (__fastcall *)(__int64))(*v81)[13];
  if ( v20 == sub_ED6D30 )
    v21 = (*(__int64 (__fastcall **)(_QWORD *))(*v81[16] + 112LL))(v81[16]);
  else
    v21 = v20((__int64)v81);
  if ( v21 )
  {
    v25 = sub_BC0510(a4, &unk_4F82418, v7);
    v26 = *(_QWORD *)(v7 + 32);
    v87 = *(_QWORD *)(v25 + 8);
    if ( v26 )
      v26 -= 56;
    v27 = sub_BC1CD0(*(_QWORD *)(v25 + 8), &unk_4F6D3F8, v26);
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    if ( (_BYTE)qword_4FE9108 )
    {
      sub_248FA70((__int64)&v110, v7, (__int64)v81, (__int64 *)(v27 + 8), v28, v29);
      v73 = v101;
      if ( v101 )
      {
        v84 = v7;
        v74 = v99;
        v75 = v99 + ((unsigned __int64)v101 << 6);
        do
        {
          if ( *(_QWORD *)v74 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            v76 = *(_QWORD **)(v74 + 24);
            while ( v76 )
            {
              v77 = (unsigned __int64)v76;
              v76 = (_QWORD *)*v76;
              j_j___libc_free_0(v77);
            }
            memset(*(void **)(v74 + 8), 0, 8LL * *(_QWORD *)(v74 + 16));
            v78 = *(_QWORD *)(v74 + 8);
            *(_QWORD *)(v74 + 32) = 0;
            *(_QWORD *)(v74 + 24) = 0;
            if ( v78 != v74 + 56 )
              j_j___libc_free_0(v78);
          }
          v74 += 64;
        }
        while ( v75 != v74 );
        v7 = v84;
        v73 = v101;
      }
      sub_C7D6A0(v99, (unsigned __int64)v73 << 6, 8);
      ++v98;
      v99 = v111;
      ++v110;
      v100 = v112;
      v111 = 0;
      v101 = (unsigned int)v113;
      v112 = 0;
      LODWORD(v113) = 0;
      sub_C7D6A0(0, 0, 8);
    }
    v106 = 0;
    v30 = *(_QWORD **)(v7 + 32);
    v107 = &v105;
    v108 = &v105;
    LODWORD(v105) = 0;
    v109 = 0;
    LODWORD(v111) = 0;
    v112 = 0;
    v113 = &v111;
    v114 = &v111;
    v115 = 0;
    if ( v88 == v30 )
    {
      v53 = 0;
      if ( !byte_4FE92E8 )
      {
LABEL_68:
        memset((void *)v4, 0, 0x60u);
        *(_BYTE *)(v4 + 28) = 1;
        *(_DWORD *)(v4 + 16) = 2;
        *(_QWORD *)(v4 + 8) = v80;
        *(_DWORD *)(v4 + 64) = 2;
        *(_QWORD *)(v4 + 56) = v79;
        *(_BYTE *)(v4 + 76) = 1;
        sub_24860C0(v53);
        sub_2485A10(v106);
        v54 = v101;
        if ( v101 )
        {
          v55 = v99;
          v56 = v99 + ((unsigned __int64)v101 << 6);
          do
          {
            if ( *(_QWORD *)v55 <= 0xFFFFFFFFFFFFFFFDLL )
            {
              v57 = *(_QWORD **)(v55 + 24);
              while ( v57 )
              {
                v58 = (unsigned __int64)v57;
                v57 = (_QWORD *)*v57;
                j_j___libc_free_0(v58);
              }
              memset(*(void **)(v55 + 8), 0, 8LL * *(_QWORD *)(v55 + 16));
              v59 = *(_QWORD *)(v55 + 8);
              *(_QWORD *)(v55 + 32) = 0;
              *(_QWORD *)(v55 + 24) = 0;
              if ( v59 != v55 + 56 )
                j_j___libc_free_0(v59);
            }
            v55 += 64;
          }
          while ( v56 != v55 );
          v54 = v101;
        }
        v12 = (__int64 *)((unsigned __int64)v54 << 6);
        sub_C7D6A0(v99, (__int64)v12, 8);
        goto LABEL_29;
      }
    }
    else
    {
      v83 = v4;
      v31 = (__int64 **)v7;
      v32 = v30;
      v33 = v31;
      do
      {
        v34 = 0;
        if ( v32 )
          v34 = v32 - 7;
        if ( !sub_B2FC80((__int64)v34) )
        {
          v35 = sub_BC1CD0(v87, &unk_4F6D3F8, (__int64)v34);
          sub_248CD60(v33, v34, (__int64)v81, (__int64 *)(v35 + 8), &v104, (__int64)&v110, (__int64)&v98);
        }
        v32 = (_QWORD *)v32[1];
      }
      while ( v88 != v32 );
      v4 = v83;
      if ( byte_4FE92E8 )
      {
        v36 = (__int64)v107;
        if ( v107 == &v105 )
        {
          v62 = v113;
        }
        else
        {
          do
          {
            v50 = sub_CB72A0();
            v51 = (_QWORD *)v50[4];
            v52 = (__int64)v50;
            if ( v50[3] - (_QWORD)v51 > 7u )
            {
              *v51 = 0x20666F72506D654DLL;
              v50[4] += 8LL;
            }
            else
            {
              v52 = sub_CB6200((__int64)v50, "MemProf ", 8u);
            }
            sub_10391D0((__int64)v102, *(_BYTE *)(v36 + 48));
            v37 = sub_CB6200(v52, v102[0], (size_t)v102[1]);
            v38 = *(__m128i **)(v37 + 32);
            v39 = v37;
            if ( *(_QWORD *)(v37 + 24) - (_QWORD)v38 <= 0x10u )
            {
              v39 = sub_CB6200(v37, " context with id ", 0x11u);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_4385460);
              v38[1].m128i_i8[0] = 32;
              *v38 = si128;
              *(_QWORD *)(v37 + 32) += 17LL;
            }
            v41 = sub_CB59D0(v39, *(_QWORD *)(v36 + 32));
            v42 = *(__m128i **)(v41 + 32);
            v43 = v41;
            if ( *(_QWORD *)(v41 + 24) - (_QWORD)v42 <= 0x18u )
            {
              v43 = sub_CB6200(v41, " has total profiled size ", 0x19u);
            }
            else
            {
              v44 = _mm_load_si128((const __m128i *)&xmmword_4385470);
              v42[1].m128i_i8[8] = 32;
              v42[1].m128i_i64[0] = 0x657A69732064656CLL;
              *v42 = v44;
              *(_QWORD *)(v41 + 32) += 25LL;
            }
            v45 = " not";
            v46 = sub_CB59D0(v43, *(_QWORD *)(v36 + 40));
            if ( *(_BYTE *)(v36 + 49) )
              v45 = " is";
            v47 = strlen(v45);
            v48 = *(_QWORD *)(v46 + 32);
            if ( *(_QWORD *)(v46 + 24) - v48 >= v47 )
            {
              if ( (_DWORD)v47 )
              {
                v60 = 0;
                do
                {
                  v61 = v60++;
                  *(_BYTE *)(v48 + v61) = v45[v61];
                }
                while ( v60 < (unsigned int)v47 );
                v48 = *(_QWORD *)(v46 + 32);
              }
              v49 = v47 + v48;
              *(_QWORD *)(v46 + 32) = v49;
            }
            else
            {
              v46 = sub_CB6200(v46, (unsigned __int8 *)v45, v47);
              v49 = *(_QWORD *)(v46 + 32);
            }
            if ( *(_QWORD *)(v46 + 24) - v49 <= 8 )
            {
              sub_CB6200(v46, " matched\n", 9u);
            }
            else
            {
              *(_BYTE *)(v49 + 8) = 10;
              *(_QWORD *)v49 = 0x6465686374616D20LL;
              *(_QWORD *)(v46 + 32) += 9LL;
            }
            if ( (__int64 *)v102[0] != &v103 )
              j_j___libc_free_0((unsigned __int64)v102[0]);
            v36 = sub_220EEE0(v36);
          }
          while ( (__int64 *)v36 != &v105 );
          v4 = v83;
          v62 = v113;
        }
        if ( v62 != &v111 )
        {
          v90 = v4;
          do
          {
            v63 = sub_CB72A0();
            v64 = (__m128i *)v63[4];
            if ( v63[3] - (_QWORD)v64 <= 0x2Bu )
            {
              sub_CB6200((__int64)v63, "MemProf callsite match for inline call stack", 0x2Cu);
            }
            else
            {
              v65 = _mm_load_si128((const __m128i *)&xmmword_4385480);
              qmemcpy(&v64[2], "e call stack", 12);
              *v64 = v65;
              v64[1] = _mm_load_si128((const __m128i *)&xmmword_4385490);
              v63[4] += 44LL;
            }
            v66 = (unsigned __int64 *)v62[5];
            for ( i = (unsigned __int64 *)v62[4]; v66 != i; ++i )
            {
              v68 = *i;
              v69 = (__int64)sub_CB72A0();
              v70 = *(_BYTE **)(v69 + 32);
              if ( *(_BYTE **)(v69 + 24) == v70 )
              {
                v69 = sub_CB6200(v69, (unsigned __int8 *)" ", 1u);
              }
              else
              {
                *v70 = 32;
                ++*(_QWORD *)(v69 + 32);
              }
              sub_CB59D0(v69, v68);
            }
            v71 = sub_CB72A0();
            v72 = (_BYTE *)v71[4];
            if ( (_BYTE *)v71[3] == v72 )
            {
              sub_CB6200((__int64)v71, (unsigned __int8 *)"\n", 1u);
            }
            else
            {
              *v72 = 10;
              ++v71[4];
            }
            v62 = (__int64 *)sub_220EF30((__int64)v62);
          }
          while ( v62 != &v111 );
          v4 = v90;
          v53 = v112;
          goto LABEL_68;
        }
      }
    }
    v53 = v112;
    goto LABEL_68;
  }
  v12 = (__int64 *)&v104;
  v110 = "Not a memory profile";
  v22 = *a2;
  LOWORD(v114) = 259;
  v106 = v22;
  v105 = 23;
  v104 = (unsigned __int64)&unk_49D9CA8;
  v107 = (__int64 *)&v110;
  sub_B6EB20((__int64)v85, (__int64)&v104);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 8) = v80;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 56) = v79;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
LABEL_29:
  ((void (__fastcall *)(_QWORD **))(*v81)[1])(v81);
LABEL_30:
  if ( (v97 & 2) != 0 )
    sub_248B690(&v96, (__int64)v12);
  if ( v96 )
    ((void (__fastcall *)(_QWORD **))(*v96)[1])(v96);
  return v4;
}
