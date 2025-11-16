// Function: sub_26A67D0
// Address: 0x26a67d0
//
__int64 __fastcall sub_26A67D0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r12
  __int64 *v10; // r8
  __int64 *v11; // r11
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r9
  __int64 v17; // rax
  _BYTE *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  bool v25; // r14
  _BYTE *v26; // r13
  __int64 *v27; // rsi
  __int64 *v28; // rcx
  __int64 v29; // rax
  bool v30; // al
  __m128i v31; // xmm1
  bool v32; // zf
  int v33; // eax
  __m128i v34; // xmm0
  __m128i v35; // xmm2
  __int64 (__fastcall *v36)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // rax
  __int64 *v37; // r15
  __int64 *v38; // r13
  unsigned __int64 i; // rax
  __int64 v40; // rdi
  unsigned int v41; // ecx
  __int64 v42; // rsi
  __int64 *v43; // r14
  __int64 *v44; // r13
  __int64 v45; // rsi
  __int64 v46; // rdi
  __int64 v48; // [rsp+50h] [rbp-9F20h]
  __int64 v49; // [rsp+50h] [rbp-9F20h]
  __int64 v51; // [rsp+70h] [rbp-9F00h]
  __int64 v52; // [rsp+78h] [rbp-9EF8h]
  __int64 v53; // [rsp+88h] [rbp-9EE8h] BYREF
  _QWORD v54[2]; // [rsp+90h] [rbp-9EE0h] BYREF
  char v55; // [rsp+A0h] [rbp-9ED0h]
  char v56[8]; // [rsp+B0h] [rbp-9EC0h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-9EB8h]
  unsigned int v58; // [rsp+C8h] [rbp-9EA8h]
  __int64 *v59; // [rsp+D0h] [rbp-9EA0h]
  __int64 v60; // [rsp+E0h] [rbp-9E90h] BYREF
  __int64 v61; // [rsp+E8h] [rbp-9E88h]
  __int64 v62; // [rsp+F0h] [rbp-9E80h]
  __int64 v63; // [rsp+F8h] [rbp-9E78h]
  __int64 *v64; // [rsp+100h] [rbp-9E70h]
  __int64 v65; // [rsp+108h] [rbp-9E68h]
  __int64 v66[2]; // [rsp+110h] [rbp-9E60h] BYREF
  __int64 *v67; // [rsp+120h] [rbp-9E50h]
  __int64 v68; // [rsp+128h] [rbp-9E48h]
  _BYTE v69[32]; // [rsp+130h] [rbp-9E40h] BYREF
  __int64 *v70; // [rsp+150h] [rbp-9E20h]
  __int64 v71; // [rsp+158h] [rbp-9E18h]
  _QWORD v72[2]; // [rsp+160h] [rbp-9E10h] BYREF
  _BYTE *v73; // [rsp+170h] [rbp-9E00h] BYREF
  __int64 v74; // [rsp+178h] [rbp-9DF8h]
  _BYTE v75[128]; // [rsp+180h] [rbp-9DF0h] BYREF
  __int64 v76; // [rsp+200h] [rbp-9D70h]
  __m128i v77; // [rsp+208h] [rbp-9D68h] BYREF
  __int64 (__fastcall *v78)(__int64 *, __m128i *, int); // [rsp+218h] [rbp-9D58h]
  __int64 (__fastcall *v79)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // [rsp+220h] [rbp-9D50h]
  _BYTE v80[16]; // [rsp+228h] [rbp-9D48h] BYREF
  __int64 (__fastcall *v81)(__int64 *, __int64); // [rsp+238h] [rbp-9D38h]
  __int64 *v82; // [rsp+240h] [rbp-9D30h]
  __int64 *v83; // [rsp+248h] [rbp-9D28h]
  __m128i *v84; // [rsp+250h] [rbp-9D20h]
  __int64 v85; // [rsp+258h] [rbp-9D18h]
  __m128i v86; // [rsp+260h] [rbp-9D10h] BYREF
  const char *v87; // [rsp+270h] [rbp-9D00h]
  _BYTE v88[16]; // [rsp+278h] [rbp-9CF8h] BYREF
  void (__fastcall *v89)(_BYTE *, _BYTE *, __int64); // [rsp+288h] [rbp-9CE8h]
  __int64 v90; // [rsp+290h] [rbp-9CE0h]
  __m128i v91; // [rsp+2A0h] [rbp-9CD0h] BYREF
  __int64 v92; // [rsp+2B0h] [rbp-9CC0h]
  __int64 (__fastcall *v93)(__int64 *, __m128i *, int); // [rsp+2B8h] [rbp-9CB8h]
  __int64 (__fastcall *v94)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // [rsp+2C0h] [rbp-9CB0h]
  _QWORD v95[2]; // [rsp+2C8h] [rbp-9CA8h] BYREF
  __int64 (__fastcall *v96)(__int64 *, __int64); // [rsp+2D8h] [rbp-9C98h]
  __int64 *v97; // [rsp+2E0h] [rbp-9C90h]
  __int64 *v98; // [rsp+2E8h] [rbp-9C88h]
  __m128i *v99; // [rsp+2F0h] [rbp-9C80h]
  __int64 v100; // [rsp+2F8h] [rbp-9C78h]
  __m128i v101; // [rsp+300h] [rbp-9C70h]
  const char *v102; // [rsp+310h] [rbp-9C60h]
  _BYTE v103[16]; // [rsp+318h] [rbp-9C58h] BYREF
  void (__fastcall *v104)(_QWORD, _QWORD, _QWORD); // [rsp+328h] [rbp-9C48h]
  __int64 v105; // [rsp+330h] [rbp-9C40h]
  __int64 v106; // [rsp+340h] [rbp-9C30h] BYREF
  char *v107; // [rsp+348h] [rbp-9C28h]
  __int64 v108; // [rsp+350h] [rbp-9C20h]
  int v109; // [rsp+358h] [rbp-9C18h]
  char v110; // [rsp+35Ch] [rbp-9C14h]
  char v111; // [rsp+360h] [rbp-9C10h] BYREF
  _BYTE *v112; // [rsp+3E0h] [rbp-9B90h]
  __int64 v113; // [rsp+3E8h] [rbp-9B88h]
  _BYTE v114[128]; // [rsp+3F0h] [rbp-9B80h] BYREF
  _BYTE *v115; // [rsp+470h] [rbp-9B00h]
  __int64 v116; // [rsp+478h] [rbp-9AF8h]
  _BYTE v117[128]; // [rsp+480h] [rbp-9AF0h] BYREF
  __int64 v118; // [rsp+500h] [rbp-9A70h]
  __int64 v119; // [rsp+508h] [rbp-9A68h]
  __int64 v120; // [rsp+510h] [rbp-9A60h]
  __int64 v121; // [rsp+518h] [rbp-9A58h]
  __int64 v122; // [rsp+520h] [rbp-9A50h]
  __m128i v123; // [rsp+530h] [rbp-9A40h] BYREF
  __int64 v124; // [rsp+540h] [rbp-9A30h]
  __int64 (__fastcall *v125)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64); // [rsp+548h] [rbp-9A28h]
  _BYTE v126[35040]; // [rsp+1690h] [rbp-88E0h] BYREF

  v9 = a1;
  v51 = a1 + 32;
  v52 = a1 + 80;
  if ( sub_2674810(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a3 + 8) + 8LL) + 40LL)) )
  {
    if ( (_BYTE)qword_4FF54C8 )
    {
      *(_BYTE *)(a1 + 76) = 1;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 8) = v51;
      *(_QWORD *)(a1 + 64) = 2;
      *(_QWORD *)(a1 + 56) = v52;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_DWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
    }
    else
    {
      v10 = *(__int64 **)(a3 + 8);
      v73 = v75;
      v74 = 0x1000000000LL;
      v11 = &v10[*(unsigned int *)(a3 + 16)];
      if ( v11 == v10 )
        goto LABEL_51;
      v12 = *v10;
      v13 = (__int64)(v10 + 1);
      v48 = a4;
      v14 = v11;
      v15 = v13;
      v16 = *(_QWORD *)(v12 + 8);
      v17 = 0;
      v18 = v75;
      v19 = v16;
      while ( 1 )
      {
        *(_QWORD *)&v18[8 * v17] = v19;
        v17 = (unsigned int)(v74 + 1);
        LODWORD(v74) = v74 + 1;
        if ( v14 == (__int64 *)v15 )
          break;
        v19 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
        if ( v17 + 1 > (unsigned __int64)HIDWORD(v74) )
        {
          sub_C8D5F0((__int64)&v73, v75, v17 + 1, 8u, v13, v16);
          v17 = (unsigned int)v74;
        }
        v18 = v73;
        v15 += 8;
      }
      v9 = a1;
      v20 = v48;
      if ( !(_DWORD)v17 )
      {
LABEL_51:
        *(_BYTE *)(v9 + 28) = 1;
        *(_QWORD *)v9 = 0;
        *(_QWORD *)(v9 + 8) = v51;
        *(_QWORD *)(v9 + 16) = 2;
        *(_DWORD *)(v9 + 24) = 0;
        *(_QWORD *)(v9 + 48) = 0;
        *(_QWORD *)(v9 + 56) = v52;
        *(_QWORD *)(v9 + 64) = 2;
        *(_DWORD *)(v9 + 72) = 0;
        *(_BYTE *)(v9 + 76) = 1;
        sub_AE6EC0(v9, (__int64)&qword_4F82400);
      }
      else
      {
        v49 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a3 + 8) + 8LL) + 40LL);
        sub_269A220((__int64)v56, v49);
        v21 = *(_QWORD *)(sub_227ED20(v20, &qword_4FDADA8, (__int64 *)a3, a5) + 8);
        v70 = v72;
        v54[0] = v21;
        v53 = v21;
        v67 = (__int64 *)v69;
        v107 = &v111;
        v68 = 0x400000000LL;
        v112 = v114;
        v113 = 0x1000000000LL;
        v115 = v117;
        v116 = 0x1000000000LL;
        v118 = a5;
        v119 = a3;
        v120 = v20;
        v54[1] = 0;
        v55 = 0;
        v66[0] = 0;
        v66[1] = 0;
        v71 = 0;
        v72[0] = 0;
        v72[1] = 1;
        v106 = 0;
        v108 = 16;
        v109 = 0;
        v110 = 1;
        v122 = 0;
        v121 = a6;
        v22 = sub_227ED20(v20, &qword_4FDADA8, (__int64 *)a3, a5);
        v23 = (unsigned __int64)v73;
        v60 = 0;
        v24 = *(_QWORD *)(v22 + 8);
        v61 = 0;
        v62 = 0;
        v122 = v24;
        LODWORD(v24) = *a2;
        v63 = 0;
        v65 = 0;
        v25 = (_DWORD)v24 == 4 || (unsigned int)(v24 - 1) <= 1;
        v26 = &v73[8 * (unsigned int)v74];
        v64 = v66;
        if ( v73 == v26 )
        {
          v28 = v66;
          v29 = -8;
        }
        else
        {
          do
          {
            v27 = (__int64 *)v23;
            v23 += 8LL;
            sub_2699F90((__int64)&v60, v27);
          }
          while ( v26 != (_BYTE *)v23 );
          v28 = v64;
          v29 = 8LL * (unsigned int)v65 - 8;
        }
        sub_2695720(
          (__int64)v126,
          *(_QWORD *)(*(__int64 *)((char *)v28 + v29) + 40),
          (__int64)v54,
          v66,
          (__int64)&v60,
          v25);
        v30 = sub_2674830(v49);
        v31 = _mm_loadu_si128(&v77);
        v32 = !v30;
        v33 = 32;
        if ( !v32 )
          v33 = qword_4FF46C8;
        v76 = 0x100000100LL;
        v81 = 0;
        LODWORD(v85) = v33;
        v86.m128i_i64[1] = (__int64)&v53;
        v87 = "openmp-opt";
        v123.m128i_i64[0] = (__int64)sub_26A1DB0;
        v34 = _mm_loadu_si128(&v123);
        v83 = &v106;
        v78 = (__int64 (__fastcall *)(__int64 *, __m128i *, int))sub_266DF20;
        v84 = 0;
        v125 = v79;
        v89 = 0;
        v79 = sub_266DF10;
        BYTE4(v85) = 1;
        v86.m128i_i64[0] = (__int64)sub_266DFD0;
        v124 = 0;
        v123 = v31;
        v77 = v34;
        sub_A17130((__int64)&v123);
        v93 = 0;
        v91.m128i_i32[0] = v76;
        v91.m128i_i16[2] = WORD2(v76);
        if ( v78 )
        {
          v78(&v91.m128i_i64[1], &v77, 2);
          v94 = v79;
          v93 = (__int64 (__fastcall *)(_QWORD *, _QWORD *, int))v78;
        }
        v96 = 0;
        if ( v81 )
        {
          ((void (__fastcall *)(_QWORD *, _BYTE *, __int64))v81)(v95, v80, 2);
          v97 = v82;
          v96 = v81;
        }
        v35 = _mm_loadu_si128(&v86);
        v104 = 0;
        v98 = v83;
        v101 = v35;
        v99 = v84;
        v100 = v85;
        v102 = v87;
        if ( v89 )
        {
          v89(v103, v88, 2);
          v105 = v90;
          v104 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v89;
        }
        sub_250EFA0((__int64)&v123, (__int64)&v60, (__int64)v126, &v91);
        sub_A17130((__int64)v103);
        sub_A17130((__int64)v95);
        sub_A17130((__int64)&v91.m128i_i64[1]);
        v91 = 0u;
        v92 = 0;
        v93 = 0;
        v36 = *(__int64 (__fastcall **)(__int64 (__fastcall **)(__int64, __int64), __int64, __int64))(*(_QWORD *)v73 + 40LL);
        v95[1] = &v106;
        v98 = (__int64 *)v126;
        v94 = v36;
        v95[0] = &v73;
        v96 = sub_266DFD0;
        v99 = &v123;
        v97 = &v53;
        if ( (_DWORD)v74 && (unsigned __int8)sub_26A50B0((__int64)&v91, 0) )
        {
          memset((void *)a1, 0, 0x60u);
          *(_BYTE *)(a1 + 28) = 1;
          *(_DWORD *)(a1 + 16) = 2;
          *(_QWORD *)(a1 + 8) = v51;
          *(_DWORD *)(a1 + 64) = 2;
          *(_QWORD *)(a1 + 56) = v52;
          *(_BYTE *)(a1 + 76) = 1;
        }
        else
        {
          *(_BYTE *)(a1 + 76) = 1;
          *(_QWORD *)(a1 + 48) = 0;
          *(_QWORD *)(a1 + 8) = v51;
          *(_QWORD *)(a1 + 64) = 2;
          *(_QWORD *)(a1 + 56) = v52;
          *(_QWORD *)(a1 + 16) = 0x100000002LL;
          *(_DWORD *)(a1 + 72) = 0;
          *(_DWORD *)(a1 + 24) = 0;
          *(_BYTE *)(a1 + 28) = 1;
          *(_QWORD *)(a1 + 32) = &qword_4F82400;
          *(_QWORD *)a1 = 1;
        }
        sub_C7D6A0(v91.m128i_i64[1], 24LL * (unsigned int)v93, 8);
        sub_250D880((__int64)&v123);
        sub_A17130((__int64)v88);
        sub_A17130((__int64)v80);
        sub_A17130((__int64)&v77);
        sub_2673E20((__int64)v126);
        if ( v64 != v66 )
          _libc_free((unsigned __int64)v64);
        sub_C7D6A0(v61, 8LL * (unsigned int)v63, 8);
        sub_29A2B10(&v106);
        if ( v115 != v117 )
          _libc_free((unsigned __int64)v115);
        if ( v112 != v114 )
          _libc_free((unsigned __int64)v112);
        if ( !v110 )
          _libc_free((unsigned __int64)v107);
        v37 = v67;
        v38 = &v67[(unsigned int)v68];
        if ( v67 != v38 )
        {
          for ( i = (unsigned __int64)v67; ; i = (unsigned __int64)v67 )
          {
            v40 = *v37;
            v41 = (unsigned int)((__int64)((__int64)v37 - i) >> 3) >> 7;
            v42 = 4096LL << v41;
            if ( v41 >= 0x1E )
              v42 = 0x40000000000LL;
            ++v37;
            sub_C7D6A0(v40, v42, 16);
            if ( v38 == v37 )
              break;
          }
        }
        v43 = v70;
        v44 = &v70[2 * (unsigned int)v71];
        if ( v70 != v44 )
        {
          do
          {
            v45 = v43[1];
            v46 = *v43;
            v43 += 2;
            sub_C7D6A0(v46, v45, 16);
          }
          while ( v44 != v43 );
          v44 = v70;
        }
        if ( v44 != v72 )
          _libc_free((unsigned __int64)v44);
        if ( v67 != (__int64 *)v69 )
          _libc_free((unsigned __int64)v67);
        if ( v59 != &v60 )
          _libc_free((unsigned __int64)v59);
        sub_C7D6A0(v57, 8LL * v58, 8);
      }
      if ( v73 != v75 )
        _libc_free((unsigned __int64)v73);
    }
  }
  else
  {
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v51;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
  }
  return v9;
}
