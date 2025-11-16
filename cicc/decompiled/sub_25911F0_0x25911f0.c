// Function: sub_25911F0
// Address: 0x25911f0
//
void __fastcall sub_25911F0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r12
  unsigned __int8 *v4; // rax
  __int64 *v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rdi
  unsigned int v8; // ebx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // r13d
  unsigned int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // ebx
  unsigned int v16; // r13d
  __int64 v17; // r14
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // r12
  __int64 i; // rbx
  int v23; // eax
  __int64 v24; // rax
  __int64 *v25; // rbx
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // rcx
  unsigned __int64 v30; // rcx
  __int64 **v31; // rdi
  _BYTE *v32; // r15
  unsigned int v33; // eax
  __int64 *v34; // r8
  __int64 v35; // r10
  int v36; // eax
  _BYTE *v37; // rdx
  __int64 **v38; // rax
  __int64 v39; // rax
  unsigned int v40; // edx
  unsigned int v41; // eax
  __m128i *v42; // rax
  __m128i *v43; // r15
  __m128i *v44; // rdx
  __m128i *v45; // rax
  __m128i *v46; // rdx
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // r14
  unsigned __int64 v49; // rdi
  int v50; // eax
  int v51; // edx
  bool v52; // zf
  const __m128i *v53; // r8
  __m128i *v54; // rax
  __m128i *v55; // rdx
  const __m128i *v56; // rax
  const __m128i *v57; // rdx
  unsigned int v58; // eax
  int v59; // r8d
  int v60; // r11d
  __int64 v61; // r10
  __int64 v62; // rbx
  unsigned int v63; // eax
  __int64 *v64; // [rsp+30h] [rbp-240h]
  __int64 *v65; // [rsp+38h] [rbp-238h]
  __int64 v66; // [rsp+48h] [rbp-228h]
  unsigned __int8 *v67; // [rsp+58h] [rbp-218h]
  __m128i *v68; // [rsp+68h] [rbp-208h]
  __int64 *v69; // [rsp+68h] [rbp-208h]
  __int64 *v70; // [rsp+70h] [rbp-200h]
  __int64 v71; // [rsp+70h] [rbp-200h]
  unsigned int v72; // [rsp+78h] [rbp-1F8h]
  char v73; // [rsp+85h] [rbp-1EBh] BYREF
  char v74; // [rsp+86h] [rbp-1EAh] BYREF
  char v75; // [rsp+87h] [rbp-1E9h] BYREF
  __int64 **v76; // [rsp+88h] [rbp-1E8h] BYREF
  __int64 *v77; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 v78; // [rsp+98h] [rbp-1D8h]
  _BYTE v79[32]; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v80; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 v81; // [rsp+C8h] [rbp-1A8h]
  __int64 v82; // [rsp+D0h] [rbp-1A0h]
  __int64 v83; // [rsp+D8h] [rbp-198h]
  __int64 **v84; // [rsp+E0h] [rbp-190h]
  __int64 v85; // [rsp+E8h] [rbp-188h]
  __int64 *v86; // [rsp+F0h] [rbp-180h] BYREF
  __int64 v87; // [rsp+F8h] [rbp-178h]
  _BYTE v88[32]; // [rsp+100h] [rbp-170h] BYREF
  void *v89; // [rsp+120h] [rbp-150h]
  void *v90; // [rsp+128h] [rbp-148h]
  unsigned int v91; // [rsp+130h] [rbp-140h]
  unsigned int v92; // [rsp+134h] [rbp-13Ch]
  char v93[8]; // [rsp+138h] [rbp-138h] BYREF
  int v94; // [rsp+140h] [rbp-130h] BYREF
  const __m128i *v95; // [rsp+148h] [rbp-128h]
  int *v96; // [rsp+150h] [rbp-120h]
  int *v97; // [rsp+158h] [rbp-118h]
  __int64 v98; // [rsp+160h] [rbp-110h]
  void *v99; // [rsp+168h] [rbp-108h]
  __int16 v100; // [rsp+170h] [rbp-100h]
  _QWORD v101[2]; // [rsp+180h] [rbp-F0h] BYREF
  unsigned __int64 v102; // [rsp+190h] [rbp-E0h]
  char v103[8]; // [rsp+198h] [rbp-D8h] BYREF
  int v104; // [rsp+1A0h] [rbp-D0h] BYREF
  unsigned __int64 v105; // [rsp+1A8h] [rbp-C8h]
  int *v106; // [rsp+1B0h] [rbp-C0h]
  int *v107; // [rsp+1B8h] [rbp-B8h]
  __int64 v108; // [rsp+1C0h] [rbp-B0h]
  void *v109; // [rsp+1C8h] [rbp-A8h]
  __int16 v110; // [rsp+1D0h] [rbp-A0h]
  __int64 v111; // [rsp+1E0h] [rbp-90h] BYREF
  void *v112; // [rsp+1E8h] [rbp-88h]
  int v113; // [rsp+1F0h] [rbp-80h]
  int v114; // [rsp+1F4h] [rbp-7Ch]
  char v115[8]; // [rsp+1F8h] [rbp-78h] BYREF
  int v116; // [rsp+200h] [rbp-70h] BYREF
  const __m128i *v117; // [rsp+208h] [rbp-68h]
  __m128i *v118; // [rsp+210h] [rbp-60h]
  const __m128i *v119; // [rsp+218h] [rbp-58h]
  __int64 v120; // [rsp+220h] [rbp-50h]

  v3 = a1 + 120;
  v68 = (__m128i *)(a1 + 72);
  v4 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  v67 = sub_BD3990(v4, (__int64)a2);
  v77 = (__int64 *)v79;
  v78 = 0x400000000LL;
  v111 = 0x5B0000005ALL;
  sub_2515D00((__int64)a2, (__m128i *)(a1 + 72), (int *)&v111, 2, (__int64)&v77, 0);
  v5 = v77;
  v70 = &v77[(unsigned int)v78];
  if ( v77 != v70 )
  {
    do
    {
      v6 = sub_A71B80(v5);
      v7 = *(_QWORD *)(a1 + 136);
      v8 = v6;
      if ( *(_DWORD *)(a1 + 108) >= v6 )
        v6 = *(_DWORD *)(a1 + 108);
      if ( *(_DWORD *)(a1 + 104) >= v8 )
        v8 = *(_DWORD *)(a1 + 104);
      v72 = v6;
      *(_DWORD *)(a1 + 108) = v6;
      v9 = v8;
      *(_DWORD *)(a1 + 104) = v8;
      if ( v7 == v3 )
      {
        v58 = v8;
      }
      else
      {
        do
        {
          v11 = *(_QWORD *)(v7 + 32);
          if ( v11 > v9 )
            break;
          v10 = *(_QWORD *)(v7 + 40) + v11;
          if ( v9 < v10 )
            v9 = v10;
          v7 = sub_220EEE0(v7);
        }
        while ( v7 != v3 );
        v58 = v9;
        if ( v8 < (unsigned int)v9 )
          v8 = v9;
      }
      *(_DWORD *)(a1 + 104) = v8;
      v12 = v72;
      if ( v72 < v58 )
        v12 = v58;
      ++v5;
      *(_DWORD *)(a1 + 108) = v12;
    }
    while ( v70 != v5 );
  }
  sub_258F340(a2, a1, v68, 1, &v73, 0, 0);
  v13 = sub_BD4FF0(v67, *(_QWORD *)(a2[26] + 104LL), &v74, &v75);
  v14 = *(_QWORD *)(a1 + 136);
  v15 = v13;
  v16 = v13;
  if ( *(_DWORD *)(a1 + 108) >= v13 )
    v16 = *(_DWORD *)(a1 + 108);
  if ( *(_DWORD *)(a1 + 104) >= v13 )
    v15 = *(_DWORD *)(a1 + 104);
  *(_DWORD *)(a1 + 108) = v16;
  *(_DWORD *)(a1 + 104) = v15;
  v17 = v15;
  v18 = v15;
  if ( v14 != v3 )
  {
    do
    {
      v20 = *(_QWORD *)(v14 + 32);
      if ( v20 > v17 )
        break;
      v19 = *(_QWORD *)(v14 + 40) + v20;
      if ( v17 < v19 )
        v17 = v19;
      v14 = sub_220EEE0(v14);
    }
    while ( v14 != v3 );
    v18 = v17;
    if ( v15 < (unsigned int)v17 )
      v15 = v17;
  }
  *(_DWORD *)(a1 + 104) = v15;
  if ( v16 < v18 )
    v16 = v18;
  *(_DWORD *)(a1 + 108) = v16;
  v21 = sub_2509740(v68);
  if ( v21 )
  {
    v66 = *(_QWORD *)(a2[26] + 120LL);
    if ( v66 )
    {
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      v84 = &v86;
      v85 = 0;
      for ( i = *(_QWORD *)(sub_250D070(v68) + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v111 = i;
        sub_25789E0((__int64)&v80, &v111);
      }
      sub_2590AF0(a1, a2, v66, v21, (__int64)&v80, a1 + 88);
      v23 = *(_DWORD *)(a1 + 108);
      if ( !v23 || *(_DWORD *)(a1 + 104) == v23 && *(_BYTE *)(a1 + 169) == *(_BYTE *)(a1 + 168) )
        goto LABEL_81;
      v86 = (__int64 *)v88;
      v87 = 0x400000000LL;
      v76 = &v86;
      sub_2568920(v66, v21, (unsigned __int8 (__fastcall *)(__int64))sub_253B5F0, (__int64)&v76);
      v64 = &v86[(unsigned int)v87];
      if ( v86 == v64 )
      {
LABEL_79:
        if ( v64 != (__int64 *)v88 )
          _libc_free((unsigned __int64)v64);
LABEL_81:
        if ( v84 != &v86 )
          _libc_free((unsigned __int64)v84);
        sub_C7D6A0(v81, 8LL * (unsigned int)v83, 8);
        goto LABEL_84;
      }
      v65 = v86;
      v71 = a1;
      while ( 1 )
      {
        v24 = *v65;
        v92 = -1;
        v89 = &unk_4A16DD8;
        v94 = 0;
        v90 = &unk_4A16D78;
        v96 = &v94;
        v97 = &v94;
        v95 = 0;
        v100 = 257;
        v98 = 0;
        v99 = &unk_4A16CD8;
        v91 = -1;
        if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
        {
          v25 = *(__int64 **)(v24 - 8);
          v26 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
          v69 = &v25[4 * v26];
          if ( v26 == 3 )
            v25 += 4;
        }
        else
        {
          v62 = v24;
          v69 = (__int64 *)v24;
          v63 = *(_DWORD *)(v24 + 4) & 0x7FFFFFF;
          v25 = (__int64 *)(v62 - 32LL * v63);
          if ( v63 == 3 )
            v25 += 4;
        }
        if ( v25 == v69 )
        {
          v51 = -1;
          *(_QWORD *)(v71 + 104) = -1;
          v50 = -1;
        }
        else
        {
          do
          {
            v27 = *v25;
            v108 = 0;
            v102 = 0xFFFFFFFF00000000LL;
            v110 = 256;
            v101[0] = &unk_4A16DD8;
            v101[1] = &unk_4A16D78;
            v107 = &v104;
            v106 = &v104;
            v28 = (unsigned int)v85;
            v104 = 0;
            v105 = 0;
            v109 = &unk_4A16CD8;
            v29 = *(_QWORD *)(v27 + 56);
            if ( v29 )
              v29 -= 24;
            sub_2590AF0(v71, a2, v66, v29, (__int64)&v80, (__int64)v101);
            v30 = (unsigned __int64)v84;
            v31 = &v84[v28];
            v32 = v31 + 1;
            if ( v31 != &v84[(unsigned int)v85] )
            {
              do
              {
                if ( (_DWORD)v83 )
                {
                  v33 = (v83 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
                  v34 = (__int64 *)(v81 + 8LL * v33);
                  v35 = *v34;
                  if ( (__int64 *)*v34 != *v31 )
                  {
                    v59 = 1;
                    if ( v35 == -4096 )
                      goto LABEL_49;
                    while ( 1 )
                    {
                      v60 = v59 + 1;
                      v33 = (v83 - 1) & (v59 + v33);
                      v34 = (__int64 *)(v81 + 8LL * v33);
                      v61 = *v34;
                      if ( *v31 == (__int64 *)*v34 )
                        break;
                      v59 = v60;
                      if ( v61 == -4096 )
                        goto LABEL_49;
                    }
                  }
                  *v34 = -8192;
                  v30 = (unsigned __int64)v84;
                  LODWORD(v82) = v82 - 1;
                  ++HIDWORD(v82);
                }
LABEL_49:
                v36 = v85;
                v37 = (_BYTE *)(v30 + 8LL * (unsigned int)v85);
                if ( v32 != v37 )
                {
                  v38 = (__int64 **)memmove(v31, v32, v37 - v32);
                  v30 = (unsigned __int64)v84;
                  v31 = v38;
                  v36 = v85;
                }
                v39 = (unsigned int)(v36 - 1);
                LODWORD(v85) = v39;
              }
              while ( v31 != (__int64 **)(v30 + 8 * v39) );
            }
            v40 = v91;
            v41 = HIDWORD(v102);
            v111 = (__int64)&unk_4A16DD8;
            if ( (unsigned int)v102 <= v91 )
              v40 = v102;
            v116 = 0;
            if ( v92 <= HIDWORD(v102) )
              v41 = v92;
            v117 = 0;
            v113 = v40;
            v92 = v41;
            v114 = v41;
            v100 &= v110;
            v91 = v40;
            v112 = &unk_4A16D78;
            v120 = 0;
            v118 = (__m128i *)&v116;
            v119 = (const __m128i *)&v116;
            if ( v95 )
            {
              v42 = sub_25394D0(v95, (__int64)&v116);
              v43 = v42;
              do
              {
                v44 = v42;
                v42 = (__m128i *)v42[1].m128i_i64[0];
              }
              while ( v42 );
              v118 = v44;
              v45 = v43;
              do
              {
                v46 = v45;
                v45 = (__m128i *)v45[1].m128i_i64[1];
              }
              while ( v45 );
              v119 = v46;
              v117 = v43;
              v120 = v98;
              v111 = (__int64)&unk_4A16DD8;
              while ( v43 )
              {
                sub_255C230((__int64)&v115, v43[1].m128i_u64[1]);
                v47 = (unsigned __int64)v43;
                v43 = (__m128i *)v43[1].m128i_i64[0];
                j_j___libc_free_0(v47);
              }
            }
            v48 = v105;
            v101[0] = &unk_4A16DD8;
            while ( v48 )
            {
              sub_255C230((__int64)&v103, *(_QWORD *)(v48 + 24));
              v49 = v48;
              v48 = *(_QWORD *)(v48 + 16);
              j_j___libc_free_0(v49);
            }
            v25 += 4;
          }
          while ( v69 != v25 );
          v50 = v91;
          v51 = v91;
          if ( *(_DWORD *)(v71 + 108) >= v91 )
            v51 = *(_DWORD *)(v71 + 108);
          if ( *(_DWORD *)(v71 + 104) >= v91 )
            v50 = *(_DWORD *)(v71 + 104);
          v52 = (_BYTE)v100 == 0;
          *(_DWORD *)(v71 + 108) = v51;
          *(_DWORD *)(v71 + 104) = v50;
          if ( v52 )
            goto LABEL_71;
        }
        *(_WORD *)(v71 + 168) = 257;
LABEL_71:
        v113 = v50;
        v114 = v51;
        v116 = 0;
        v112 = &unk_4A16D78;
        v111 = (__int64)&unk_4A16DD8;
        v53 = *(const __m128i **)(v71 + 128);
        v117 = 0;
        v118 = (__m128i *)&v116;
        v119 = (const __m128i *)&v116;
        v120 = 0;
        if ( v53 )
        {
          v54 = sub_25394D0(v53, (__int64)&v116);
          v53 = v54;
          do
          {
            v55 = v54;
            v54 = (__m128i *)v54[1].m128i_i64[0];
          }
          while ( v54 );
          v118 = v55;
          v56 = v53;
          do
          {
            v57 = v56;
            v56 = (const __m128i *)v56[1].m128i_i64[1];
          }
          while ( v56 );
          v119 = v57;
          v117 = v53;
          v120 = *(_QWORD *)(v71 + 152);
        }
        v111 = (__int64)&unk_4A16DD8;
        sub_255C230((__int64)&v115, (unsigned __int64)v53);
        v89 = &unk_4A16DD8;
        sub_255C230((__int64)&v93, (unsigned __int64)v95);
        if ( v64 == ++v65 )
        {
          v64 = v86;
          goto LABEL_79;
        }
      }
    }
  }
LABEL_84:
  if ( v77 != (__int64 *)v79 )
    _libc_free((unsigned __int64)v77);
}
