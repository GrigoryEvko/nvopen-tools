// Function: sub_2E38390
// Address: 0x2e38390
//
unsigned __int64 __fastcall sub_2E38390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rbx
  _WORD *v8; // rdx
  __int64 (*v9)(void); // rax
  _BYTE *v10; // rax
  __int64 v11; // rax
  void *v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // r15
  __int64 v15; // rsi
  void **v16; // rdi
  __int64 v17; // rdx
  _BYTE *v18; // rax
  __int64 v19; // r15
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdi
  _BYTE *v25; // rax
  _WORD *v26; // rdx
  unsigned __int64 result; // rax
  char v28; // r12
  __int64 v29; // r13
  unsigned __int64 v30; // r14
  unsigned int v31; // edx
  __int64 v32; // rsi
  unsigned int v33; // r8d
  unsigned int v34; // eax
  __int64 v35; // rcx
  unsigned __int64 v36; // rdi
  unsigned __int64 j; // rax
  __int64 k; // r9
  __int16 v39; // cx
  unsigned int v40; // edi
  __int64 *v41; // rcx
  __int64 v42; // r9
  _BYTE *v43; // rax
  _BYTE *v44; // rax
  __m128i *v45; // rdx
  _BYTE *v46; // rax
  _BYTE *v47; // rax
  _BYTE *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r15
  unsigned int *v52; // r12
  _WORD *v53; // rdx
  __int64 v54; // rax
  _WORD *v55; // rdx
  _WORD *v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // r15
  _QWORD *v59; // rax
  _BYTE *v60; // rax
  __int64 v61; // rax
  char *v62; // rdx
  __int64 v63; // rdi
  char *v64; // rax
  bool v65; // cf
  _BYTE *v66; // rax
  __int64 v67; // rax
  __m128i *v68; // rdx
  __int64 v69; // rdi
  __m128i si128; // xmm0
  int v71; // edi
  _WORD *v72; // rdx
  __int64 *v73; // r13
  int i; // eax
  __int64 v75; // rbx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  _BYTE *v79; // rax
  __int64 v80; // rdx
  double v81; // xmm0_8
  double v82; // xmm1_8
  __int64 v83; // rdi
  _BYTE *v84; // rax
  int v85; // r12d
  _WORD *v86; // rdx
  __int64 v87; // rdi
  __int64 *v88; // r12
  char v89; // bl
  __int64 v90; // r15
  __int64 v91; // r9
  _WORD *v92; // rdx
  __int64 v93; // rax
  _BYTE *v94; // rax
  __m128i *v95; // rdx
  __int64 v96; // rdx
  int v97; // ecx
  __int64 v98; // rax
  int v99; // r10d
  __int64 *v100; // [rsp+0h] [rbp-100h]
  __int64 v102; // [rsp+8h] [rbp-F8h]
  __int64 v103; // [rsp+8h] [rbp-F8h]
  __int64 v104; // [rsp+8h] [rbp-F8h]
  __int64 v105; // [rsp+10h] [rbp-F0h]
  __int64 v106; // [rsp+10h] [rbp-F0h]
  __int64 v109; // [rsp+28h] [rbp-D8h]
  __int64 v111; // [rsp+38h] [rbp-C8h]
  __int64 *v112; // [rsp+40h] [rbp-C0h]
  unsigned int *v113; // [rsp+40h] [rbp-C0h]
  char v114; // [rsp+40h] [rbp-C0h]
  __int64 *v115; // [rsp+40h] [rbp-C0h]
  _QWORD v117[2]; // [rsp+50h] [rbp-B0h] BYREF
  void (__fastcall *v118)(_QWORD *, _QWORD *, __int64); // [rsp+60h] [rbp-A0h]
  void (__fastcall *v119)(_QWORD *, __int64); // [rsp+68h] [rbp-98h]
  _QWORD v120[2]; // [rsp+70h] [rbp-90h] BYREF
  void (__fastcall *v121)(_QWORD *, _QWORD *, __int64); // [rsp+80h] [rbp-80h]
  void (__fastcall *v122)(_QWORD *, __int64); // [rsp+88h] [rbp-78h]
  _QWORD v123[2]; // [rsp+90h] [rbp-70h] BYREF
  void (__fastcall *v124)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-60h]
  void (__fastcall *v125)(_QWORD *, __int64); // [rsp+A8h] [rbp-58h]
  void *v126; // [rsp+B0h] [rbp-50h] BYREF
  char *v127; // [rsp+B8h] [rbp-48h]
  double v128; // [rsp+C0h] [rbp-40h]
  __int64 (__fastcall *v129)(__int64 *, __int64); // [rsp+C8h] [rbp-38h]

  v5 = a2;
  v6 = *(_QWORD *)(a1 + 32);
  if ( !v6 )
  {
    v95 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v95 <= 0x3Fu )
    {
      v98 = sub_CB6200(a2, "Can't print out MachineBasicBlock because parent MachineFunction", 0x40u);
      v96 = *(_QWORD *)(v98 + 32);
      v5 = v98;
    }
    else
    {
      *v95 = _mm_load_si128((const __m128i *)&xmmword_42E9C20);
      v95[1] = _mm_load_si128((const __m128i *)&xmmword_42E9C30);
      v95[2] = _mm_load_si128((const __m128i *)&xmmword_42E9C40);
      v95[3] = _mm_load_si128((const __m128i *)&xmmword_42E9C50);
      v96 = *(_QWORD *)(a2 + 32) + 64LL;
      *(_QWORD *)(a2 + 32) = v96;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v96) <= 8 )
      return sub_CB6200(v5, " is null\n", 9u);
    *(_BYTE *)(v96 + 8) = 10;
    *(_QWORD *)v96 = 0x6C6C756E20736920LL;
    *(_QWORD *)(v5 + 32) += 9LL;
    return 0x6C6C756E20736920LL;
  }
  v7 = a1;
  if ( a4 && (_BYTE)qword_501EBC8 )
  {
    v126 = *(void **)(*(_QWORD *)(a4 + 152) + 16LL * *(unsigned int *)(a1 + 24));
    sub_2FAD600(&v126, a2);
    v46 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v46 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 9);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v46 + 1;
      *v46 = 9;
    }
  }
  sub_2E37380(a1, a2, 3, a3);
  v8 = *(_WORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(a2, (unsigned __int8 *)":\n", 2u);
  }
  else
  {
    *v8 = 2618;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  v111 = 0;
  v109 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v6 + 16) + 200LL))(*(_QWORD *)(v6 + 16));
  v105 = *(_QWORD *)(v6 + 32);
  v9 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 128LL);
  if ( v9 != sub_2DAC790 )
    v111 = v9();
  if ( *(_DWORD *)(a1 + 72) && a5 )
  {
    if ( a4 )
    {
      v44 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v44 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 9);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v44 + 1;
        *v44 = 9;
      }
    }
    v45 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v45 <= 0xFu )
    {
      sub_CB6200(a2, "; predecessors: ", 0x10u);
    }
    else
    {
      *v45 = _mm_load_si128((const __m128i *)&xmmword_42E9C70);
      *(_QWORD *)(a2 + 32) += 16LL;
    }
    v88 = *(__int64 **)(a1 + 64);
    v115 = &v88[*(unsigned int *)(a1 + 72)];
    if ( v115 != v88 )
    {
      v89 = 1;
      do
      {
        v91 = *v88;
        if ( v89 )
        {
          v90 = v5;
          v89 = 0;
        }
        else
        {
          v92 = *(_WORD **)(v5 + 32);
          if ( *(_QWORD *)(v5 + 24) - (_QWORD)v92 > 1u )
          {
            v90 = v5;
            *v92 = 8236;
            *(_QWORD *)(v5 + 32) += 2LL;
          }
          else
          {
            v104 = *v88;
            v93 = sub_CB6200(v5, (unsigned __int8 *)", ", 2u);
            v91 = v104;
            v90 = v93;
          }
        }
        v15 = v91;
        v16 = (void **)v117;
        sub_2E31000(v117, v91);
        if ( !v118 )
          goto LABEL_189;
        v119(v117, v90);
        if ( v118 )
          v118(v117, v117, 3);
        ++v88;
      }
      while ( v115 != v88 );
      v7 = a1;
    }
    v94 = *(_BYTE **)(v5 + 32);
    if ( (unsigned __int64)v94 >= *(_QWORD *)(v5 + 24) )
    {
      sub_CB5D20(v5, 10);
      if ( !*(_DWORD *)(v7 + 120) )
        goto LABEL_69;
    }
    else
    {
      *(_QWORD *)(v5 + 32) = v94 + 1;
      *v94 = 10;
      if ( !*(_DWORD *)(v7 + 120) )
      {
        if ( *(_QWORD *)(v7 + 192) == *(_QWORD *)(v7 + 184) )
          goto LABEL_101;
        goto LABEL_70;
      }
    }
  }
  else if ( !*(_DWORD *)(a1 + 120) )
  {
    if ( *(_QWORD *)(a1 + 184) == *(_QWORD *)(a1 + 192) || (*(_BYTE *)(*(_QWORD *)v105 + 344LL) & 4) == 0 )
      goto LABEL_30;
    goto LABEL_71;
  }
  if ( a4 )
  {
    v10 = *(_BYTE **)(v5 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v5 + 24) )
    {
      sub_CB5D20(v5, 9);
    }
    else
    {
      *(_QWORD *)(v5 + 32) = v10 + 1;
      *v10 = 9;
    }
  }
  v11 = sub_CB69B0(v5, 2u);
  v12 = *(void **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xBu )
  {
    sub_CB6200(v11, "successors: ", 0xCu);
  }
  else
  {
    qmemcpy(v12, "successors: ", 12);
    *(_QWORD *)(v11 + 32) += 12LL;
  }
  v13 = *(__int64 **)(v7 + 112);
  v14 = v5;
  v112 = &v13[*(unsigned int *)(v7 + 120)];
  if ( v112 != v13 )
  {
    while ( 1 )
    {
      v15 = *v13;
      v16 = (void **)v120;
      sub_2E31000(v120, *v13);
      if ( !v121 )
        break;
      v122(v120, v14);
      if ( v121 )
        v121(v120, v120, 3);
      if ( *(_QWORD *)(v7 + 152) != *(_QWORD *)(v7 + 144) )
      {
        v18 = *(_BYTE **)(v5 + 32);
        if ( (unsigned __int64)v18 >= *(_QWORD *)(v5 + 24) )
        {
          v19 = sub_CB5D20(v5, 40);
        }
        else
        {
          v19 = v5;
          *(_QWORD *)(v5 + 32) = v18 + 1;
          *v18 = 40;
        }
        v20 = sub_2E32EA0(v7, (__int64)v13);
        v127 = "0x%08x";
        LODWORD(v128) = v20;
        v126 = &unk_49DD0F8;
        v24 = sub_CB6620(v19, (__int64)&v126, (__int64)&unk_49DD0F8, v21, v22, v23);
        v25 = *(_BYTE **)(v24 + 32);
        if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 24) )
        {
          sub_CB5D20(v24, 41);
        }
        else
        {
          *(_QWORD *)(v24 + 32) = v25 + 1;
          *v25 = 41;
        }
      }
      if ( ++v13 == v112 )
        goto LABEL_65;
      v26 = *(_WORD **)(v5 + 32);
      if ( *(_QWORD *)(v5 + 24) - (_QWORD)v26 > 1u )
      {
        v14 = v5;
        *v26 = 8236;
        *(_QWORD *)(v5 + 32) += 2LL;
      }
      else
      {
        v14 = sub_CB6200(v5, (unsigned __int8 *)", ", 2u);
      }
    }
LABEL_189:
    sub_4263D6(v16, v15, v17);
  }
LABEL_65:
  if ( *(_QWORD *)(v7 + 152) != *(_QWORD *)(v7 + 144) && a5 )
  {
    v72 = *(_WORD **)(v5 + 32);
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v72 <= 1u )
    {
      sub_CB6200(v5, (unsigned __int8 *)"; ", 2u);
    }
    else
    {
      *v72 = 8251;
      *(_QWORD *)(v5 + 32) += 2LL;
    }
    v73 = *(__int64 **)(v7 + 112);
    v100 = &v73[*(unsigned int *)(v7 + 120)];
    if ( v100 != v73 )
    {
      v114 = 1;
      v103 = v7;
      for ( i = sub_2E32EA0(v7, (__int64)v73); ; i = sub_2E32EA0(v103, (__int64)v73) )
      {
        v85 = i;
        if ( v114 )
        {
          v114 = 0;
          v75 = v5;
        }
        else
        {
          v86 = *(_WORD **)(v5 + 32);
          if ( *(_QWORD *)(v5 + 24) - (_QWORD)v86 > 1u )
          {
            v75 = v5;
            *v86 = 8236;
            *(_QWORD *)(v5 + 32) += 2LL;
          }
          else
          {
            v75 = sub_CB6200(v5, (unsigned __int8 *)", ", 2u);
          }
        }
        v15 = *v73;
        v16 = (void **)v123;
        sub_2E31000(v123, *v73);
        if ( !v124 )
          goto LABEL_189;
        v125(v123, v75);
        v79 = *(_BYTE **)(v75 + 32);
        if ( (unsigned __int64)v79 >= *(_QWORD *)(v75 + 24) )
        {
          v75 = sub_CB5D20(v75, 40);
        }
        else
        {
          v80 = (__int64)(v79 + 1);
          *(_QWORD *)(v75 + 32) = v79 + 1;
          *v79 = 40;
        }
        v81 = (double)v85 * 4.656612873077393e-10 * 100.0 * 100.0;
        v82 = fabs(v81);
        if ( v82 < 4.503599627370496e15 )
          *(_QWORD *)&v81 = COERCE_UNSIGNED_INT64(v82 + 4.503599627370496e15 - 4.503599627370496e15)
                          | *(_QWORD *)&v81 & 0x8000000000000000LL;
        v127 = "%.2f%%";
        v128 = v81 / 100.0;
        v126 = &unk_49DD0B8;
        v83 = sub_CB6620(v75, (__int64)&v126, v80, v76, v77, v78);
        v84 = *(_BYTE **)(v83 + 32);
        if ( (unsigned __int64)v84 >= *(_QWORD *)(v83 + 24) )
        {
          sub_CB5D20(v83, 41);
        }
        else
        {
          *(_QWORD *)(v83 + 32) = v84 + 1;
          *v84 = 41;
        }
        if ( v124 )
          v124(v123, v123, 3);
        if ( ++v73 == v100 )
          break;
      }
      v7 = v103;
    }
  }
  v47 = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)v47 >= *(_QWORD *)(v5 + 24) )
  {
    sub_CB5D20(v5, 10);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v47 + 1;
    *v47 = 10;
  }
LABEL_69:
  if ( *(_QWORD *)(v7 + 184) == *(_QWORD *)(v7 + 192) )
    goto LABEL_101;
LABEL_70:
  if ( (*(_BYTE *)(*(_QWORD *)v105 + 344LL) & 4) != 0 )
  {
LABEL_71:
    if ( a4 )
    {
      v48 = *(_BYTE **)(v5 + 32);
      if ( (unsigned __int64)v48 >= *(_QWORD *)(v5 + 24) )
      {
        sub_CB5D20(v5, 9);
      }
      else
      {
        *(_QWORD *)(v5 + 32) = v48 + 1;
        *v48 = 9;
      }
    }
    v49 = sub_CB69B0(v5, 2u);
    v50 = *(_QWORD *)(v49 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v49 + 24) - v50) <= 8 )
    {
      sub_CB6200(v49, "liveins: ", 9u);
    }
    else
    {
      *(_BYTE *)(v50 + 8) = 32;
      *(_QWORD *)v50 = 0x3A736E696576696CLL;
      *(_QWORD *)(v49 + 32) += 9LL;
    }
    v51 = v5;
    v113 = *(unsigned int **)(v7 + 192);
    v52 = (unsigned int *)sub_2E33140(v7);
    if ( v113 != v52 )
    {
      while ( 1 )
      {
        v15 = *v52;
        v16 = &v126;
        sub_2FF6320(&v126, v15, v109, 0, 0);
        if ( v128 == 0.0 )
          goto LABEL_189;
        v129((__int64 *)&v126, v51);
        if ( v128 != 0.0 )
          (*(void (__fastcall **)(void **, void **, __int64))&v128)(&v126, &v126, 3);
        if ( *((_QWORD *)v52 + 1) == -1 && *((_QWORD *)v52 + 2) == -1 )
          goto LABEL_82;
        v57 = *(_QWORD *)(v5 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v57) <= 2 )
        {
          v58 = sub_CB6200(v5, ":0x", 3u);
        }
        else
        {
          *(_BYTE *)(v57 + 2) = 120;
          *(_WORD *)v57 = 12346;
          v58 = v5;
          *(_QWORD *)(v5 + 32) += 3LL;
        }
        v102 = *((_QWORD *)v52 + 1);
        v106 = *((_QWORD *)v52 + 2);
        v59 = (_QWORD *)sub_22077B0(0x10u);
        if ( v59 )
        {
          *v59 = v102;
          v59[1] = v106;
        }
        v126 = v59;
        v128 = COERCE_DOUBLE(sub_2E09350);
        v129 = sub_2E092F0;
        sub_2E092F0((__int64 *)&v126, v58);
        if ( v128 == 0.0 )
        {
LABEL_82:
          v52 += 6;
          if ( v113 == v52 )
            break;
        }
        else
        {
          v52 += 6;
          (*(void (__fastcall **)(void **, void **, __int64))&v128)(&v126, &v126, 3);
          if ( v113 == v52 )
            break;
        }
        v53 = *(_WORD **)(v5 + 32);
        if ( *(_QWORD *)(v5 + 24) - (_QWORD)v53 > 1u )
        {
          v51 = v5;
          *v53 = 8236;
          *(_QWORD *)(v5 + 32) += 2LL;
        }
        else
        {
          v51 = sub_CB6200(v5, (unsigned __int8 *)", ", 2u);
        }
      }
    }
  }
LABEL_101:
  v60 = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)v60 >= *(_QWORD *)(v5 + 24) )
  {
    sub_CB5D20(v5, 10);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v60 + 1;
    *v60 = 10;
  }
LABEL_30:
  result = v7 + 48;
  v28 = 0;
  if ( v7 + 48 == *(_QWORD *)(v7 + 56) )
    goto LABEL_55;
  v29 = v5;
  v30 = *(_QWORD *)(v7 + 56);
  do
  {
    if ( a4 && (_BYTE)qword_501EBC8 )
    {
      v31 = *(_DWORD *)(a4 + 144);
      v32 = *(_QWORD *)(a4 + 128);
      if ( v31 )
      {
        v33 = v31 - 1;
        v34 = (v31 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v35 = *(_QWORD *)(v32 + 16LL * v34);
        if ( v35 == v30 )
        {
LABEL_36:
          v36 = v30;
          for ( j = v30; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
            ;
          if ( (*(_DWORD *)(v30 + 44) & 8) != 0 )
          {
            do
              v36 = *(_QWORD *)(v36 + 8);
            while ( (*(_BYTE *)(v36 + 44) & 8) != 0 );
          }
          for ( k = *(_QWORD *)(v36 + 8); k != j; j = *(_QWORD *)(j + 8) )
          {
            v39 = *(_WORD *)(j + 68);
            if ( (unsigned __int16)(v39 - 14) > 4u && v39 != 24 )
              break;
          }
          v40 = v33 & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
          v41 = (__int64 *)(v32 + 16LL * v40);
          v42 = *v41;
          if ( *v41 != j )
          {
            v97 = 1;
            while ( v42 != -4096 )
            {
              v99 = v97 + 1;
              v40 = v33 & (v97 + v40);
              v41 = (__int64 *)(v32 + 16LL * v40);
              v42 = *v41;
              if ( j == *v41 )
                goto LABEL_45;
              v97 = v99;
            }
            v41 = (__int64 *)(v32 + 16LL * v31);
          }
LABEL_45:
          v126 = (void *)v41[1];
          sub_2FAD600(&v126, v29);
        }
        else
        {
          v71 = 1;
          while ( v35 != -4096 )
          {
            v34 = v33 & (v71 + v34);
            v35 = *(_QWORD *)(v32 + 16LL * v34);
            if ( v35 == v30 )
              goto LABEL_36;
            ++v71;
          }
        }
      }
      v43 = *(_BYTE **)(v29 + 32);
      if ( (unsigned __int64)v43 >= *(_QWORD *)(v29 + 24) )
      {
        sub_CB5D20(v29, 9);
      }
      else
      {
        *(_QWORD *)(v29 + 32) = v43 + 1;
        *v43 = 9;
      }
    }
    if ( v28 )
    {
      if ( (*(_BYTE *)(v30 + 44) & 4) != 0 )
      {
        sub_CB69B0(v29, 4u);
        sub_2E8BCA0(v30, v29, a3, a5, 0, 0, 0, v111);
        result = *(_QWORD *)(v29 + 32);
        goto LABEL_51;
      }
      v54 = sub_CB69B0(v29, 2u);
      v55 = *(_WORD **)(v54 + 32);
      if ( *(_QWORD *)(v54 + 24) - (_QWORD)v55 <= 1u )
      {
        sub_CB6200(v54, "}\n", 2u);
      }
      else
      {
        *v55 = 2685;
        *(_QWORD *)(v54 + 32) += 2LL;
      }
    }
    sub_CB69B0(v29, 2u);
    sub_2E8BCA0(v30, v29, a3, a5, 0, 0, 0, v111);
    if ( (*(_BYTE *)(v30 + 44) & 8) != 0 )
    {
      v56 = *(_WORD **)(v29 + 32);
      if ( *(_QWORD *)(v29 + 24) - (_QWORD)v56 <= 1u )
      {
        sub_CB6200(v29, (unsigned __int8 *)" {", 2u);
        result = *(_QWORD *)(v29 + 32);
      }
      else
      {
        *v56 = 31520;
        result = *(_QWORD *)(v29 + 32) + 2LL;
        *(_QWORD *)(v29 + 32) = result;
      }
      v28 = 1;
LABEL_51:
      if ( *(_QWORD *)(v29 + 24) > result )
        goto LABEL_52;
      goto LABEL_88;
    }
    result = *(_QWORD *)(v29 + 32);
    v28 = 0;
    if ( *(_QWORD *)(v29 + 24) > result )
    {
LABEL_52:
      *(_QWORD *)(v29 + 32) = result + 1;
      *(_BYTE *)result = 10;
      goto LABEL_53;
    }
LABEL_88:
    result = sub_CB5D20(v29, 10);
LABEL_53:
    v30 = *(_QWORD *)(v30 + 8);
  }
  while ( v7 + 48 != v30 );
  v5 = v29;
  if ( v28 )
  {
    v61 = sub_CB69B0(v29, 2u);
    v62 = *(char **)(v61 + 32);
    v63 = v61;
    v64 = *(char **)(v61 + 24);
    v65 = v64 == v62;
    result = v64 - v62;
    if ( v65 || result == 1 )
    {
      result = sub_CB6200(v63, "}\n", 2u);
    }
    else
    {
      *(_WORD *)v62 = 2685;
      *(_QWORD *)(v63 + 32) += 2LL;
    }
  }
LABEL_55:
  if ( *(_BYTE *)(v7 + 176) && a5 )
  {
    if ( a4 )
    {
      v66 = *(_BYTE **)(v5 + 32);
      if ( (unsigned __int64)v66 >= *(_QWORD *)(v5 + 24) )
      {
        sub_CB5D20(v5, 9);
      }
      else
      {
        *(_QWORD *)(v5 + 32) = v66 + 1;
        *v66 = 9;
      }
    }
    v67 = sub_CB69B0(v5, 2u);
    v68 = *(__m128i **)(v67 + 32);
    v69 = v67;
    if ( *(_QWORD *)(v67 + 24) - (_QWORD)v68 <= 0x21u )
    {
      v69 = sub_CB6200(v67, "; Irreducible loop header weight: ", 0x22u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42E9C80);
      v68[2].m128i_i16[0] = 8250;
      *v68 = si128;
      v68[1] = _mm_load_si128((const __m128i *)&xmmword_42E9C90);
      *(_QWORD *)(v67 + 32) += 34LL;
    }
    v87 = sub_CB59D0(v69, *(_QWORD *)(v7 + 168));
    result = *(_QWORD *)(v87 + 32);
    if ( result >= *(_QWORD *)(v87 + 24) )
    {
      return sub_CB5D20(v87, 10);
    }
    else
    {
      *(_QWORD *)(v87 + 32) = result + 1;
      *(_BYTE *)result = 10;
    }
  }
  return result;
}
