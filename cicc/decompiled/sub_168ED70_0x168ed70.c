// Function: sub_168ED70
// Address: 0x168ed70
//
void __fastcall sub_168ED70(__int64 a1)
{
  _QWORD *v2; // rdx
  _BYTE *v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // rdi
  _QWORD *v6; // r11
  _QWORD *v7; // r8
  _QWORD *v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 (__fastcall ***v11)(__int64); // r14
  __int64 (__fastcall ***v12)(__int64); // r13
  __m128i *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  __int64 v15; // rax
  __int64 (__fastcall **v16)(__int64); // rdi
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  size_t v20; // r15
  unsigned int v21; // r9d
  __int64 v22; // rdx
  __int64 (__fastcall ***v23)(__int64); // r14
  __int64 (__fastcall ***v24)(__int64); // r13
  __m128i *v25; // rdi
  __int64 (__fastcall *v26)(__int64); // rax
  __int64 (__fastcall **v27)(__int64); // rdi
  unsigned __int64 v28; // r8
  __int64 v29; // r12
  __int64 v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  unsigned int v33; // r9d
  _QWORD *v34; // r10
  _QWORD *v35; // rcx
  unsigned int v36; // eax
  __int64 *v37; // rax
  __int64 *v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *v41; // rdi
  _QWORD *v42; // r15
  unsigned int i; // eax
  unsigned int v44; // r13d
  __int64 v45; // rdi
  __int64 v46; // rdi
  char v47; // al
  const __m128i *v48; // rbx
  char *v49; // rax
  unsigned int v50; // r15d
  __int64 v51; // rdx
  __int64 v52; // rsi
  _BYTE *v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // rdi
  _BYTE *v56; // r13
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 (__fastcall *v59)(__int64, _BYTE *, int); // rax
  __m128i v60; // xmm1
  bool v61; // zf
  int v62; // eax
  __int64 v63; // rdx
  __int64 *v64; // rdi
  int v65; // eax
  unsigned __int64 v66; // rax
  bool v67; // cc
  int v68; // eax
  _QWORD *v69; // [rsp+8h] [rbp-238h]
  _BYTE *v70; // [rsp+18h] [rbp-228h]
  char *v71; // [rsp+20h] [rbp-220h]
  _QWORD *v72; // [rsp+28h] [rbp-218h]
  _QWORD *v73; // [rsp+30h] [rbp-210h]
  __int64 v74; // [rsp+30h] [rbp-210h]
  _QWORD *v75; // [rsp+38h] [rbp-208h]
  const __m128i *v76; // [rsp+38h] [rbp-208h]
  _QWORD *v77; // [rsp+40h] [rbp-200h]
  _QWORD *v78; // [rsp+40h] [rbp-200h]
  _BYTE *v79; // [rsp+40h] [rbp-200h]
  unsigned int v80; // [rsp+4Ch] [rbp-1F4h]
  unsigned int v81; // [rsp+4Ch] [rbp-1F4h]
  bool v82; // [rsp+4Ch] [rbp-1F4h]
  void *src; // [rsp+50h] [rbp-1F0h]
  _QWORD *v84; // [rsp+58h] [rbp-1E8h]
  __m128i v85; // [rsp+60h] [rbp-1E0h] BYREF
  _QWORD v86[2]; // [rsp+70h] [rbp-1D0h] BYREF
  __int16 v87; // [rsp+80h] [rbp-1C0h]
  _QWORD v88[2]; // [rsp+90h] [rbp-1B0h] BYREF
  __int16 v89; // [rsp+A0h] [rbp-1A0h]
  unsigned __int64 v90; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v91; // [rsp+B8h] [rbp-188h]
  __int64 v92; // [rsp+C0h] [rbp-180h]
  __int64 v93; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v94; // [rsp+D8h] [rbp-168h]
  __int64 v95; // [rsp+E0h] [rbp-160h]
  int v96; // [rsp+E8h] [rbp-158h]
  __m128i v97; // [rsp+F0h] [rbp-150h] BYREF
  _BYTE *v98; // [rsp+100h] [rbp-140h] BYREF
  _QWORD *v99; // [rsp+108h] [rbp-138h]
  _QWORD *v100; // [rsp+110h] [rbp-130h]
  _QWORD *v101; // [rsp+118h] [rbp-128h]
  _QWORD *v102; // [rsp+120h] [rbp-120h]
  _QWORD *v103; // [rsp+128h] [rbp-118h]
  _BYTE *v104; // [rsp+130h] [rbp-110h] BYREF
  size_t n; // [rsp+138h] [rbp-108h]
  _BYTE v106[64]; // [rsp+140h] [rbp-100h] BYREF
  __int64 (__fastcall **v107)(__int64); // [rsp+180h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+188h] [rbp-B8h]
  __int64 (__fastcall *v109)(__int64); // [rsp+190h] [rbp-B0h] BYREF
  __int64 v110; // [rsp+198h] [rbp-A8h]
  __int64 (__fastcall *v111)(__int64); // [rsp+1A0h] [rbp-A0h]
  __int64 v112; // [rsp+1A8h] [rbp-98h]
  __int64 (__fastcall *v113)(__int64 *); // [rsp+1B0h] [rbp-90h]
  __int64 v114; // [rsp+1B8h] [rbp-88h]

  v92 = 0x1000000000LL;
  v2 = *(_QWORD **)(a1 + 264);
  v104 = v106;
  v3 = (_BYTE *)v2[6];
  v4 = v2 + 7;
  v5 = v2 + 5;
  v6 = v2 + 1;
  v7 = v2 + 3;
  v95 = 0;
  v96 = 0;
  v8 = (_QWORD *)v2[2];
  n = 0x4000000000LL;
  v9 = (_QWORD *)v2[4];
  v10 = v2[8];
  v97.m128i_i64[1] = (__int64)v4;
  v90 = 0;
  v91 = 0;
  v93 = 0;
  v94 = 0;
  v70 = v5;
  v75 = v6;
  v84 = v7;
  v97.m128i_i64[0] = v10;
  v98 = v3;
  v99 = v5;
  v100 = v8;
  v101 = v6;
  v102 = v9;
  v103 = v7;
  if ( v7 == v9 )
    goto LABEL_19;
  do
  {
    do
    {
      v11 = &v107;
      v110 = 0;
      v12 = &v107;
      v13 = &v97;
      v109 = sub_168DD90;
      v112 = 0;
      v111 = sub_168DDB0;
      v114 = 0;
      v113 = sub_168DDD0;
      v14 = sub_168DD70;
      if ( ((unsigned __int8)sub_168DD70 & 1) == 0 )
        goto LABEL_4;
      while ( 1 )
      {
        v14 = *(__int64 (__fastcall **)(__int64))((char *)v14 + v13->m128i_i64[0] - 1);
LABEL_4:
        v15 = v14((__int64)v13);
        if ( v15 )
          break;
        while ( 1 )
        {
          v16 = v12[3];
          v14 = (__int64 (__fastcall *)(__int64))v12[2];
          v11 += 2;
          v12 = v11;
          v13 = (__m128i *)((char *)&v97 + (_QWORD)v16);
          if ( ((unsigned __int8)v14 & 1) != 0 )
            break;
          v15 = v14((__int64)v13);
          if ( v15 )
            goto LABEL_7;
        }
      }
LABEL_7:
      v17 = v15;
      if ( (*(_BYTE *)(v15 + 23) & 0x20) == 0 )
        goto LABEL_13;
      LODWORD(n) = 0;
      sub_1649960(v15);
      v19 = v18 + 1;
      if ( v19 > HIDWORD(n) )
        sub_16CD150(&v104, v106, v19, 1);
      sub_38BA670(&v93, &v104, v17, 0);
      v20 = (unsigned int)n;
      src = v104;
      v21 = sub_16D19C0(&v90, v104, (unsigned int)n);
      v22 = *(_QWORD *)(v90 + 8LL * v21);
      if ( !v22 )
        goto LABEL_38;
      if ( v22 == -8 )
      {
        LODWORD(v92) = v92 - 1;
LABEL_38:
        v77 = (_QWORD *)(v90 + 8LL * v21);
        v80 = v21;
        v32 = malloc(v20 + 17);
        v33 = v80;
        v34 = v77;
        v35 = (_QWORD *)v32;
        if ( !v32 )
        {
          sub_16BD1C0("Allocation failed");
          v35 = 0;
          v34 = v77;
          v33 = v80;
        }
        if ( v20 )
        {
          v73 = v35;
          v78 = v34;
          v81 = v33;
          memcpy(v35 + 2, src, v20);
          v35 = v73;
          v34 = v78;
          v33 = v81;
        }
        *((_BYTE *)v35 + v20 + 16) = 0;
        *v35 = v20;
        v35[1] = 0;
        *v34 = v35;
        ++HIDWORD(v91);
        v36 = sub_16D1CD0(&v90, v33);
        v37 = (__int64 *)(v90 + 8LL * v36);
        v22 = *v37;
        if ( *v37 == -8 || !v22 )
        {
          v38 = v37 + 1;
          do
          {
            do
              v22 = *v38++;
            while ( v22 == -8 );
          }
          while ( !v22 );
        }
      }
      *(_QWORD *)(v22 + 8) = v17;
LABEL_13:
      v23 = &v107;
      v110 = 0;
      v112 = 0;
      v24 = &v107;
      v25 = &v97;
      v109 = sub_168DCE0;
      v114 = 0;
      v111 = sub_168DD10;
      v113 = sub_168DD40;
      v26 = sub_168DCB0;
      if ( ((unsigned __int8)sub_168DCB0 & 1) == 0 )
        goto LABEL_15;
      while ( 1 )
      {
        v26 = *(__int64 (__fastcall **)(__int64))((char *)v26 + v25->m128i_i64[0] - 1);
LABEL_15:
        if ( (unsigned __int8)v26((__int64)v25) )
          break;
        while ( 1 )
        {
          v27 = v24[3];
          v26 = (__int64 (__fastcall *)(__int64))v24[2];
          v23 += 2;
          v24 = v23;
          v25 = (__m128i *)((char *)&v97 + (_QWORD)v27);
          if ( ((unsigned __int8)v26 & 1) != 0 )
            break;
          if ( (unsigned __int8)v26((__int64)v25) )
            goto LABEL_18;
        }
      }
LABEL_18:
      ;
    }
    while ( v84 != v102 );
LABEL_19:
    ;
  }
  while ( v84 != v103
       || v75 != v100
       || v75 != v101
       || v70 != v98
       || v70 != (_BYTE *)v99
       || v4 != (_QWORD *)v97.m128i_i64[0]
       || v4 != (_QWORD *)v97.m128i_i64[1] );
  if ( *(_DWORD *)(a1 + 320) )
  {
    v39 = *(_QWORD **)(a1 + 312);
    v40 = 4LL * *(unsigned int *)(a1 + 328);
    v41 = &v39[v40];
    v72 = &v39[v40];
    if ( v39 != &v39[v40] )
    {
      while ( 1 )
      {
        v42 = v39;
        if ( *v39 != -16 && *v39 != -8 )
          break;
        v39 += 4;
        if ( v41 == v39 )
          goto LABEL_27;
      }
      v79 = (_BYTE *)*v39;
      if ( v72 != v39 )
      {
        for ( i = sub_168E790(a1, v79); ; i = sub_168E790(a1, (_BYTE *)*v42) )
        {
          if ( i == 4 )
          {
            v82 = 1;
            v44 = 20;
            goto LABEL_67;
          }
          if ( i > 4 )
          {
            v61 = i == 6;
            v62 = 20;
            v82 = 0;
            if ( !v61 )
              v62 = 0;
            v44 = v62;
          }
          else
          {
            v44 = i & 0xFFFFFFFD;
            if ( (i & 0xFFFFFFFD) != 0 )
            {
              if ( i - 2 <= 2 )
              {
                v82 = 1;
                v44 = 8;
                goto LABEL_67;
              }
              v82 = 0;
              v44 = 8;
            }
            else
            {
              v82 = i - 2 <= 2;
            }
          }
          v45 = *(_QWORD *)(a1 + 264);
          if ( (*v79 & 4) != 0 )
          {
            v46 = sub_1632000(v45, *((_QWORD *)v79 - 1) + 16LL, **((_QWORD **)v79 - 1));
            if ( v46 )
              goto LABEL_61;
          }
          else
          {
            v46 = sub_1632000(v45, 0, 0);
            if ( v46 )
              goto LABEL_61;
          }
          v63 = 0;
          if ( (*v79 & 4) != 0 )
          {
            v64 = (__int64 *)*((_QWORD *)v79 - 1);
            v63 = *v64;
            v46 = (__int64)(v64 + 2);
          }
          v65 = sub_16D1B30(&v90, v46, v63);
          if ( v65 == -1 )
            goto LABEL_67;
          v66 = v90 + 8LL * v65;
          if ( v66 == v90 + 8LL * (unsigned int)v91 )
            goto LABEL_67;
          v46 = *(_QWORD *)(*(_QWORD *)v66 + 8LL);
          if ( !v46 )
            goto LABEL_67;
LABEL_61:
          if ( !v44 )
          {
            v47 = *(_BYTE *)(v46 + 32) & 0xF;
            if ( !v47 )
            {
              v44 = 8;
              if ( v82 )
                goto LABEL_67;
              goto LABEL_66;
            }
            v44 = 13;
            if ( ((v47 + 9) & 0xFu) > 1 )
            {
              v44 = 20;
              if ( ((v47 + 14) & 0xFu) > 3 )
              {
                v67 = ((v47 + 7) & 0xFu) <= 1;
                v68 = 20;
                if ( !v67 )
                  v68 = 0;
                v44 = v68;
              }
            }
          }
          if ( v82 || (*(_BYTE *)(v46 + 32) & 0xF) == 1 )
            goto LABEL_67;
LABEL_66:
          v82 = !sub_15E4F60(v46);
LABEL_67:
          v48 = (const __m128i *)v42[1];
          v76 = (const __m128i *)v42[2];
          if ( v76 != v48 )
          {
            v49 = "@";
            v69 = v42;
            if ( v82 )
              v49 = "@@";
            v50 = v44;
            v71 = v49;
            do
            {
              v85 = _mm_loadu_si128(v48);
              v54 = sub_16D20C0(&v85, word_3F645A0, 3, 0);
              if ( v54 == -1 )
              {
                v60 = _mm_loadu_si128(&v85);
                v98 = 0;
                v99 = 0;
                v107 = &v109;
                v108 = 0x8000000000LL;
                v97 = v60;
              }
              else
              {
                v51 = v54 + 3;
                if ( v54 + 3 > v85.m128i_i64[1] )
                  v51 = v85.m128i_i64[1];
                v52 = v85.m128i_i64[1] - v51;
                v53 = (_BYTE *)(v85.m128i_i64[0] + v51);
                if ( v54 && v54 > v85.m128i_i64[1] )
                  v54 = v85.m128i_u64[1];
                v97.m128i_i64[1] = v54;
                v97.m128i_i64[0] = v85.m128i_i64[0];
                v107 = &v109;
                v98 = v53;
                v99 = (_QWORD *)v52;
                v108 = 0x8000000000LL;
                if ( v52 && *v53 != 64 )
                {
                  v89 = 1282;
                  v86[0] = &v97;
                  v87 = 773;
                  v86[1] = v71;
                  v88[0] = v86;
                  v88[1] = &v98;
                  sub_16E2F40(v88, &v107);
                  v85.m128i_i64[0] = (__int64)v107;
                  v85.m128i_i64[1] = (unsigned int)v108;
                }
              }
              v55 = *(_QWORD *)(a1 + 8);
              v88[0] = &v85;
              v89 = 261;
              v56 = (_BYTE *)sub_38BF510(v55, v88);
              v57 = sub_38CF310(v79, 0, *(_QWORD *)(a1 + 8), 0);
              v58 = v57;
              if ( v82 )
              {
                v74 = v57;
                sub_168E000(a1, v56);
                v58 = v74;
              }
              sub_38DDC10(a1, v56, v58);
              if ( v50 )
              {
                v59 = *(__int64 (__fastcall **)(__int64, _BYTE *, int))(*(_QWORD *)a1 + 256LL);
                if ( v59 == sub_168E630 )
                {
                  if ( v50 == 8 || v50 == 20 )
                    sub_168E230(a1, v56, v50);
                }
                else
                {
                  v59(a1, v56, v50);
                }
              }
              if ( v107 != &v109 )
                _libc_free((unsigned __int64)v107);
              ++v48;
            }
            while ( v76 != v48 );
            v42 = v69;
          }
          v42 += 4;
          if ( v42 == v72 )
            break;
          while ( *v42 == -16 || *v42 == -8 )
          {
            v42 += 4;
            if ( v72 == v42 )
              goto LABEL_27;
          }
          if ( v72 == v42 )
            break;
          v79 = (_BYTE *)*v42;
        }
      }
    }
  }
LABEL_27:
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  j___libc_free_0(v94);
  if ( HIDWORD(v91) )
  {
    v28 = v90;
    if ( (_DWORD)v91 )
    {
      v29 = 8LL * (unsigned int)v91;
      v30 = 0;
      do
      {
        v31 = *(_QWORD *)(v28 + v30);
        if ( v31 != -8 && v31 )
        {
          _libc_free(v31);
          v28 = v90;
        }
        v30 += 8;
      }
      while ( v29 != v30 );
    }
  }
  else
  {
    v28 = v90;
  }
  _libc_free(v28);
}
