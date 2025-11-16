// Function: sub_2DB7EF0
// Address: 0x2db7ef0
//
__int64 __fastcall sub_2DB7EF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 (*v3)(); // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned int v26; // ebx
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned int v30; // r12d
  __int64 v31; // rbx
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 *v34; // rcx
  __int64 *v35; // rdx
  __int64 v36; // rdx
  _QWORD *v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // r13
  unsigned int v42; // r12d
  unsigned int v43; // r14d
  unsigned int v44; // eax
  __int64 v45; // r14
  __int64 v46; // r13
  unsigned int v47; // ebx
  unsigned int v48; // eax
  __int64 v49; // rdi
  __int64 (*v50)(); // rax
  unsigned __int64 v51; // rcx
  __int64 *v52; // rbx
  __int64 v53; // rdi
  __int64 *v54; // r14
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // r12
  __int64 v58; // rcx
  __int64 *v59; // rbx
  __int64 j; // r15
  __int64 v61; // rsi
  _QWORD *v62; // rdi
  _BYTE *v63; // rax
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rcx
  bool v67; // zf
  unsigned int v68; // r13d
  unsigned int v69; // r12d
  __int64 v70; // r14
  unsigned int v71; // eax
  __int64 v72; // rdi
  __int64 (*v73)(); // rax
  unsigned __int8 v74; // bl
  __int64 *v75; // rax
  __m128i *v77; // rdx
  __int64 v78; // rsi
  const __m128i *v79; // rax
  __int64 v80; // rcx
  __m128i *v81; // rdx
  const __m128i *v82; // rax
  const __m128i *v83; // rcx
  __int64 v84; // [rsp-8h] [rbp-5A8h]
  _QWORD *v86; // [rsp+28h] [rbp-578h]
  unsigned __int8 v87; // [rsp+37h] [rbp-569h]
  unsigned __int8 i; // [rsp+38h] [rbp-568h]
  _QWORD *v89; // [rsp+38h] [rbp-568h]
  unsigned int v90; // [rsp+40h] [rbp-560h]
  __int64 v91; // [rsp+48h] [rbp-558h]
  __int64 *v92; // [rsp+50h] [rbp-550h]
  __int64 v93; // [rsp+58h] [rbp-548h]
  unsigned int v94; // [rsp+58h] [rbp-548h]
  unsigned __int8 v95; // [rsp+58h] [rbp-548h]
  __int64 v96; // [rsp+68h] [rbp-538h]
  __int64 v97; // [rsp+68h] [rbp-538h]
  __int64 *v98; // [rsp+68h] [rbp-538h]
  __int64 v99; // [rsp+68h] [rbp-538h]
  __int64 v100; // [rsp+78h] [rbp-528h] BYREF
  __int64 *v101; // [rsp+80h] [rbp-520h] BYREF
  __int64 v102; // [rsp+88h] [rbp-518h]
  _BYTE v103[32]; // [rsp+90h] [rbp-510h] BYREF
  char v104[8]; // [rsp+B0h] [rbp-4F0h] BYREF
  unsigned __int64 v105; // [rsp+B8h] [rbp-4E8h]
  char v106; // [rsp+CCh] [rbp-4D4h]
  char v107[64]; // [rsp+D0h] [rbp-4D0h] BYREF
  __m128i *v108; // [rsp+110h] [rbp-490h] BYREF
  __int64 v109; // [rsp+118h] [rbp-488h]
  _BYTE v110[192]; // [rsp+120h] [rbp-480h] BYREF
  char v111[8]; // [rsp+1E0h] [rbp-3C0h] BYREF
  unsigned __int64 v112; // [rsp+1E8h] [rbp-3B8h]
  char v113; // [rsp+1FCh] [rbp-3A4h]
  char v114[64]; // [rsp+200h] [rbp-3A0h] BYREF
  __m128i *v115; // [rsp+240h] [rbp-360h] BYREF
  __int64 v116; // [rsp+248h] [rbp-358h]
  _BYTE v117[192]; // [rsp+250h] [rbp-350h] BYREF
  char v118[8]; // [rsp+310h] [rbp-290h] BYREF
  unsigned __int64 v119; // [rsp+318h] [rbp-288h]
  char v120; // [rsp+32Ch] [rbp-274h]
  const __m128i *v121; // [rsp+370h] [rbp-230h]
  unsigned int v122; // [rsp+378h] [rbp-228h]
  char v123; // [rsp+380h] [rbp-220h] BYREF
  char v124[8]; // [rsp+440h] [rbp-160h] BYREF
  unsigned __int64 v125; // [rsp+448h] [rbp-158h]
  char v126; // [rsp+45Ch] [rbp-144h]
  const __m128i *v127; // [rsp+4A0h] [rbp-100h]
  unsigned int v128; // [rsp+4A8h] [rbp-F8h]
  char v129; // [rsp+4B0h] [rbp-F0h] BYREF

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = ((__int64 (__fastcall *)(_QWORD))v3)(*(_QWORD *)(a2 + 16));
  a1[25] = v4;
  a1[26] = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 200LL))(v2);
  a1[64] = *(_QWORD *)(a2 + 32);
  v86 = a1 + 27;
  sub_2FF7BB0(a1 + 27, v2);
  v5 = (__int64 *)a1[1];
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_148:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_501FE44 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_148;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_501FE44);
  v9 = (__int64 *)a1[1];
  a1[65] = v8 + 200;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_149:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_50208AC )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_149;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_50208AC);
  v13 = (__int64 *)a1[1];
  a1[67] = v12 + 200;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_150:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_501F1C8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_150;
  }
  a1[66] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
             *(_QWORD *)(v14 + 8),
             &unk_501F1C8)
         + 169;
  v92 = a1 + 68;
  sub_2DB2850((__int64)(a1 + 68), a2);
  sub_2DB5AF0((__int64)v118, a1[65], v16, v17, v18, v19);
  sub_C8CD80((__int64)v104, (__int64)v107, (__int64)v118, v20, v21, v22);
  v26 = v122;
  v108 = (__m128i *)v110;
  v109 = 0x800000000LL;
  if ( v122 )
  {
    v77 = (__m128i *)v110;
    v78 = v122;
    if ( v122 > 8 )
    {
      sub_2DB5930((__int64)&v108, v122, (__int64)v110, v23, v24, v25);
      v77 = v108;
      v78 = v122;
    }
    v79 = v121;
    v23 = (__int64)&v121->m128i_i64[3 * v78];
    if ( v121 != (const __m128i *)v23 )
    {
      do
      {
        if ( v77 )
        {
          *v77 = _mm_loadu_si128(v79);
          v77[1].m128i_i64[0] = v79[1].m128i_i64[0];
        }
        v79 = (const __m128i *)((char *)v79 + 24);
        v77 = (__m128i *)((char *)v77 + 24);
      }
      while ( (const __m128i *)v23 != v79 );
    }
    LODWORD(v109) = v26;
  }
  sub_C8CD80((__int64)v111, (__int64)v114, (__int64)v124, v23, v24, v25);
  v30 = v128;
  v115 = (__m128i *)v117;
  v116 = 0x800000000LL;
  if ( v128 )
  {
    v31 = v128;
    v80 = v128;
    if ( v128 > 8 )
    {
      sub_2DB5930((__int64)&v115, v128, v27, v128, v28, v29);
      v81 = v115;
      v80 = v128;
    }
    else
    {
      v81 = (__m128i *)v117;
    }
    v82 = v127;
    v83 = (const __m128i *)((char *)v127 + 24 * v80);
    if ( v127 != v83 )
    {
      do
      {
        if ( v81 )
        {
          *v81 = _mm_loadu_si128(v82);
          v81[1].m128i_i64[0] = v82[1].m128i_i64[0];
        }
        v82 = (const __m128i *)((char *)v82 + 24);
        v81 = (__m128i *)((char *)v81 + 24);
      }
      while ( v83 != v82 );
    }
    LODWORD(v116) = v30;
  }
  else
  {
    v31 = 0;
  }
  v87 = 0;
  v32 = v109;
  while ( 1 )
  {
    v33 = 24LL * v32;
    if ( v32 != v31 )
      goto LABEL_23;
    v29 = (__int64)v108->m128i_i64 + v33;
    v28 = (__int64)v115;
    if ( v108 == (__m128i *)&v108->m128i_i8[v33] )
      break;
    v34 = (__int64 *)v115;
    v35 = (__int64 *)v108;
    while ( v35[2] == v34[2] && v35[1] == v34[1] && *v35 == *v34 )
    {
      v35 += 3;
      v34 += 3;
      if ( (__int64 *)v29 == v35 )
        goto LABEL_108;
    }
LABEL_23:
    v36 = *(__int64 *)((char *)&v108->m128i_i64[-1] + v33);
    v37 = a1;
    v38 = *(_QWORD *)v36;
    a1[72] = 0;
    a1[74] = 0;
    a1[71] = v38;
    a1[73] = 0;
    v91 = v38;
    if ( *(_DWORD *)(v38 + 120) == 2 )
    {
      for ( i = 0; ; i = v95 )
      {
        v33 = v91;
        if ( !(unsigned __int8)sub_2DB49F0(v92, v91, 1) )
        {
LABEL_85:
          v66 = i;
          v74 = v87;
          v32 = v109;
          if ( i )
            v74 = i;
          v95 = v74;
          goto LABEL_88;
        }
        v33 = v37[71];
        v90 = sub_2E441D0(v37[66], v33, v37[73]);
        v93 = v37[73];
        v39 = v93;
        v40 = v37[72];
        if ( v93 == v40 )
        {
          v93 = v37[74];
        }
        else
        {
          v28 = v37[74];
          if ( v40 != v28 )
          {
            v41 = *(_QWORD *)(v93 + 56);
            v42 = 0;
            v94 = 0;
            if ( v39 + 48 != v41 )
            {
              v96 = v39 + 48;
              v43 = 0;
              do
              {
                while ( 1 )
                {
                  v44 = sub_2FF8080(v86, v41, 0);
                  v33 = v41;
                  if ( v44 > 1 )
                    v42 = v44 + v42 - 1;
                  v43 += (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v37[25] + 1168LL))(v37[25], v41);
                  if ( !v41 )
                    BUG();
                  if ( (*(_BYTE *)v41 & 4) == 0 )
                    break;
                  v41 = *(_QWORD *)(v41 + 8);
                  if ( v96 == v41 )
                    goto LABEL_36;
                }
                while ( (*(_BYTE *)(v41 + 44) & 8) != 0 )
                  v41 = *(_QWORD *)(v41 + 8);
                v41 = *(_QWORD *)(v41 + 8);
              }
              while ( v96 != v41 );
LABEL_36:
              v94 = v43;
              v28 = v37[74];
            }
            v45 = *(_QWORD *)(v28 + 56);
            v46 = 0;
            v47 = 0;
            v97 = v28 + 48;
            if ( v45 != v28 + 48 )
            {
              do
              {
                while ( 1 )
                {
                  v48 = sub_2FF8080(v86, v45, 0);
                  v33 = v45;
                  if ( v48 > 1 )
                    v47 = v48 + v47 - 1;
                  v46 = (*(unsigned int (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v37[25] + 1168LL))(v37[25], v45)
                      + (unsigned int)v46;
                  if ( !v45 )
                    BUG();
                  if ( (*(_BYTE *)v45 & 4) == 0 )
                    break;
                  v45 = *(_QWORD *)(v45 + 8);
                  if ( v45 == v97 )
                    goto LABEL_45;
                }
                while ( (*(_BYTE *)(v45 + 44) & 8) != 0 )
                  v45 = *(_QWORD *)(v45 + 8);
                v45 = *(_QWORD *)(v45 + 8);
              }
              while ( v45 != v97 );
LABEL_45:
              v28 = v37[74];
            }
            v49 = v37[25];
            v50 = *(__int64 (**)())(*(_QWORD *)v49 + 424LL);
            if ( v50 == sub_2DB1B00 )
              goto LABEL_85;
            v33 = v37[73];
            v95 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, __int64, _QWORD, __int64, _QWORD))v50)(
                    v49,
                    v33,
                    v42,
                    v94,
                    v28,
                    v47,
                    v46,
                    v90);
            v36 = v84;
            goto LABEL_48;
          }
        }
        v68 = 0;
        v69 = 0;
        v70 = *(_QWORD *)(v93 + 56);
        if ( v70 != v93 + 48 )
        {
          v99 = v93 + 48;
          do
          {
            while ( 1 )
            {
              v71 = sub_2FF8080(v86, v70, 0);
              v33 = v70;
              if ( v71 > 1 )
                v69 = v71 + v69 - 1;
              v68 += (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v37[25] + 1168LL))(v37[25], v70);
              if ( !v70 )
                BUG();
              if ( (*(_BYTE *)v70 & 4) == 0 )
                break;
              v70 = *(_QWORD *)(v70 + 8);
              if ( v99 == v70 )
                goto LABEL_84;
            }
            while ( (*(_BYTE *)(v70 + 44) & 8) != 0 )
              v70 = *(_QWORD *)(v70 + 8);
            v70 = *(_QWORD *)(v70 + 8);
          }
          while ( v99 != v70 );
        }
LABEL_84:
        v72 = v37[25];
        v73 = *(__int64 (**)())(*(_QWORD *)v72 + 416LL);
        if ( v73 == sub_2DB1AF0 )
          goto LABEL_85;
        v33 = v93;
        v95 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD))v73)(v72, v93, v69, v68, v90);
LABEL_48:
        if ( !v95 )
          goto LABEL_85;
        v101 = (__int64 *)v103;
        v102 = 0x400000000LL;
        sub_2DB3620(v92, &v101, 1, v51, v28, v29);
        v33 = v37[71];
        sub_2DB2600(v37[65], v33, v101, (unsigned int)v102);
        v52 = v101;
        v98 = &v101[(unsigned int)v102];
        if ( v101 != v98 )
        {
          do
          {
            v53 = *v52++;
            sub_2E32710(v53);
          }
          while ( v98 != v52 );
          v54 = v101;
          v55 = v37[67];
          v98 = &v101[(unsigned int)v102];
          if ( v98 != v101 )
          {
            v89 = v37;
            do
            {
              v56 = *(unsigned int *)(v55 + 24);
              v57 = *v54;
              v58 = *(_QWORD *)(v55 + 8);
              if ( (_DWORD)v56 )
              {
                v36 = ((_DWORD)v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
                v59 = (__int64 *)(v58 + 16 * v36);
                v33 = *v59;
                if ( v57 == *v59 )
                {
LABEL_55:
                  if ( v59 != (__int64 *)(v58 + 16 * v56) )
                  {
                    for ( j = v59[1]; j; j = *(_QWORD *)j )
                    {
                      v61 = *(_QWORD *)(j + 40);
                      v62 = *(_QWORD **)(j + 32);
                      v100 = v57;
                      v63 = sub_2DB1D70(v62, v61, &v100);
                      sub_2DB58F0(j + 32, v63);
                      v28 = v100;
                      if ( *(_BYTE *)(j + 84) )
                      {
                        v33 = *(_QWORD *)(j + 64);
                        v64 = v33 + 8LL * *(unsigned int *)(j + 76);
                        v36 = v33;
                        if ( v33 != v64 )
                        {
                          while ( v100 != *(_QWORD *)v36 )
                          {
                            v36 += 8;
                            if ( v64 == v36 )
                              goto LABEL_63;
                          }
                          v65 = (unsigned int)(*(_DWORD *)(j + 76) - 1);
                          *(_DWORD *)(j + 76) = v65;
                          *(_QWORD *)v36 = *(_QWORD *)(v33 + 8 * v65);
                          ++*(_QWORD *)(j + 56);
                        }
                      }
                      else
                      {
                        v33 = v100;
                        v75 = sub_C8CA60(j + 56, v100);
                        if ( v75 )
                        {
                          *v75 = -2;
                          ++*(_DWORD *)(j + 80);
                          ++*(_QWORD *)(j + 56);
                        }
                      }
LABEL_63:
                      ;
                    }
                    *v59 = -8192;
                    --*(_DWORD *)(v55 + 16);
                    ++*(_DWORD *)(v55 + 20);
                  }
                }
                else
                {
                  v28 = 1;
                  while ( v33 != -4096 )
                  {
                    v29 = (unsigned int)(v28 + 1);
                    v36 = ((_DWORD)v56 - 1) & (unsigned int)(v28 + v36);
                    v59 = (__int64 *)(v58 + 16LL * (unsigned int)v36);
                    v33 = *v59;
                    if ( v57 == *v59 )
                      goto LABEL_55;
                    v28 = (unsigned int)v29;
                  }
                }
              }
              ++v54;
            }
            while ( v98 != v54 );
            v37 = v89;
            v98 = v101;
          }
        }
        v66 = (__int64)v103;
        if ( v98 != (__int64 *)v103 )
          _libc_free((unsigned __int64)v98);
        v37[72] = 0;
        v37[74] = 0;
        v37[71] = v91;
        v37[73] = 0;
        if ( *(_DWORD *)(v91 + 120) != 2 )
        {
          v67 = (_DWORD)v109 == 1;
          v32 = v109 - 1;
          LODWORD(v109) = v109 - 1;
          if ( v67 )
            goto LABEL_89;
          goto LABEL_71;
        }
      }
    }
    v66 = v87;
    v95 = v87;
LABEL_88:
    LODWORD(v109) = --v32;
    if ( v32 )
    {
LABEL_71:
      sub_2DB5710((__int64)v104, v33, v36, v66, v28, v29);
      v32 = v109;
    }
LABEL_89:
    v31 = (unsigned int)v116;
    v87 = v95;
  }
LABEL_108:
  if ( v115 != (__m128i *)v117 )
    _libc_free((unsigned __int64)v115);
  if ( !v113 )
    _libc_free(v112);
  if ( v108 != (__m128i *)v110 )
    _libc_free((unsigned __int64)v108);
  if ( !v106 )
    _libc_free(v105);
  if ( v127 != (const __m128i *)&v129 )
    _libc_free((unsigned __int64)v127);
  if ( !v126 )
    _libc_free(v125);
  if ( v121 != (const __m128i *)&v123 )
    _libc_free((unsigned __int64)v121);
  if ( !v120 )
    _libc_free(v119);
  return v87;
}
