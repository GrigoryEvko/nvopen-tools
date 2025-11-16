// Function: sub_2F542E0
// Address: 0x2f542e0
//
__int64 __fastcall sub_2F542E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rdi
  unsigned __int64 v9; // rsi
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 (*v16)(); // rdx
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 (*v19)(); // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rdi
  __int64 *v24; // r12
  __int64 (__fastcall *v25)(__int64, __int64); // rax
  _DWORD *v26; // r13
  int v27; // eax
  __int64 *v28; // r13
  __int64 v29; // r15
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 (__fastcall *v33)(__int64); // rax
  __int64 v34; // rsi
  __int64 (__fastcall *v35)(__int64); // rax
  int v36; // r10d
  unsigned int *v37; // r8
  unsigned int *v38; // rdi
  unsigned int *v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rsi
  char v42; // cl
  unsigned __int16 v43; // ax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  __int64 v47; // r14
  __int64 *v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rdi
  unsigned int *v51; // rax
  __int64 v52; // r13
  _DWORD *i; // r12
  _DWORD *v54; // rdx
  unsigned __int64 v55; // rcx
  __int64 v56; // rbx
  unsigned int v57; // edx
  __int64 v58; // rbx
  bool v59; // zf
  _QWORD **v60; // rsi
  _QWORD **v61; // rdx
  _QWORD **v62; // rax
  __int64 v63; // rcx
  __int64 v65; // rcx
  __int64 v66; // rsi
  int v67; // eax
  _DWORD *v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 v72; // r9
  __int64 v73; // rax
  unsigned int *v74; // rax
  _DWORD *v75; // rdx
  __int64 *v76; // rax
  __int64 *v77; // rax
  __int64 v78; // r8
  __int64 v79; // r9
  unsigned __int64 v80; // r10
  _DWORD *v81; // rdx
  unsigned __int64 v82; // rcx
  _DWORD *v83; // rax
  _DWORD *v84; // rcx
  __int64 v85; // [rsp+0h] [rbp-220h]
  __int64 v86; // [rsp+8h] [rbp-218h]
  __int64 *v87; // [rsp+8h] [rbp-218h]
  int v88; // [rsp+10h] [rbp-210h]
  __int64 v89; // [rsp+10h] [rbp-210h]
  _QWORD *v90; // [rsp+20h] [rbp-200h]
  __int64 v91; // [rsp+28h] [rbp-1F8h]
  __int64 v92; // [rsp+30h] [rbp-1F0h]
  __int64 v93; // [rsp+38h] [rbp-1E8h]
  unsigned int *v94; // [rsp+38h] [rbp-1E8h]
  __int64 v95; // [rsp+40h] [rbp-1E0h]
  unsigned __int64 v96; // [rsp+40h] [rbp-1E0h]
  int v97; // [rsp+48h] [rbp-1D8h]
  char v98; // [rsp+4Ch] [rbp-1D4h]
  int v99; // [rsp+4Ch] [rbp-1D4h]
  __int64 *v100; // [rsp+50h] [rbp-1D0h]
  int v101; // [rsp+50h] [rbp-1D0h]
  _QWORD v103[2]; // [rsp+70h] [rbp-1B0h] BYREF
  char v104; // [rsp+80h] [rbp-1A0h]
  unsigned int *v105; // [rsp+90h] [rbp-190h] BYREF
  __int64 v106; // [rsp+98h] [rbp-188h]
  _BYTE v107[128]; // [rsp+A0h] [rbp-180h] BYREF
  _QWORD v108[2]; // [rsp+120h] [rbp-100h] BYREF
  __int64 v109; // [rsp+130h] [rbp-F0h]
  __int64 v110; // [rsp+138h] [rbp-E8h]
  __int64 v111; // [rsp+140h] [rbp-E0h]
  __int64 v112; // [rsp+148h] [rbp-D8h]
  __int64 v113; // [rsp+150h] [rbp-D0h]
  __int64 v114; // [rsp+158h] [rbp-C8h]
  unsigned int v115; // [rsp+160h] [rbp-C0h]
  char v116; // [rsp+164h] [rbp-BCh]
  __int64 v117; // [rsp+168h] [rbp-B8h]
  __int64 v118; // [rsp+170h] [rbp-B0h]
  char *v119; // [rsp+178h] [rbp-A8h]
  __int64 v120; // [rsp+180h] [rbp-A0h]
  int v121; // [rsp+188h] [rbp-98h]
  char v122; // [rsp+18Ch] [rbp-94h]
  char v123; // [rsp+190h] [rbp-90h] BYREF
  __int64 v124; // [rsp+1B0h] [rbp-70h]
  char *v125; // [rsp+1B8h] [rbp-68h]
  __int64 v126; // [rsp+1C0h] [rbp-60h]
  int v127; // [rsp+1C8h] [rbp-58h]
  char v128; // [rsp+1CCh] [rbp-54h]
  char v129; // [rsp+1D0h] [rbp-50h] BYREF

  v6 = a1;
  v7 = a1 + 48;
  v91 = v7;
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 - 32) + 56LL) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF))
     & 0xFFFFFFFFFFFFFFF8LL;
  v95 = v9;
  v10 = *(_QWORD *)(v6 + 48) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v9 + 24LL);
  if ( *(_DWORD *)(v6 + 56) != *(_DWORD *)v10 )
    sub_2F60630(v7, v9, 3LL * *(unsigned __int16 *)(*(_QWORD *)v9 + 24LL), a4);
  v98 = *(_BYTE *)(v10 + 8);
  if ( !v98 && !*(_QWORD *)(a2 + 104) )
    return 0;
  v11 = *(_QWORD *)(v6 + 24);
  v12 = *(_QWORD *)(v6 + 32);
  v109 = a4;
  v13 = *(_QWORD *)(v6 + 768);
  v108[0] = &unk_4A388F0;
  v108[1] = a2;
  v14 = *(_QWORD *)(v13 + 32);
  v112 = v11;
  v110 = v14;
  v111 = v12;
  v15 = *(_QWORD *)(v13 + 16);
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 128LL);
  v17 = 0;
  if ( v16 != sub_2DAC790 )
  {
    v17 = ((__int64 (__fastcall *)(__int64))v16)(v15);
    v14 = v110;
  }
  v113 = v17;
  v18 = *(_DWORD *)(a4 + 8);
  v114 = v6 + 760;
  v115 = v18;
  v119 = &v123;
  v116 = 0;
  v117 = v6 + 400;
  v118 = 0;
  v120 = 4;
  v121 = 0;
  v122 = 1;
  v124 = 0;
  v125 = &v129;
  v126 = 4;
  v127 = 0;
  v128 = 1;
  if ( !*(_BYTE *)(v14 + 36) )
    goto LABEL_86;
  v19 = *(__int64 (**)())(v14 + 16);
  v12 = *(unsigned int *)(v14 + 28);
  v16 = (__int64 (*)())((char *)v19 + 8 * v12);
  if ( v19 != v16 )
  {
    while ( *(_QWORD **)v19 != v108 )
    {
      v19 = (__int64 (*)())((char *)v19 + 8);
      if ( v16 == v19 )
        goto LABEL_85;
    }
    goto LABEL_12;
  }
LABEL_85:
  if ( (unsigned int)v12 < *(_DWORD *)(v14 + 24) )
  {
    *(_DWORD *)(v14 + 28) = v12 + 1;
    *(_QWORD *)v16 = v108;
    ++*(_QWORD *)(v14 + 8);
  }
  else
  {
LABEL_86:
    sub_C8CC70(v14 + 8, (__int64)v108, (__int64)v16, v12, v15, a6);
  }
LABEL_12:
  sub_2FB3410(*(_QWORD *)(v6 + 1000), v108, 1);
  v21 = *(_QWORD *)(v6 + 992);
  v22 = *(unsigned int *)(v21 + 208);
  if ( v22 <= 1 )
    goto LABEL_54;
  v23 = *(_QWORD *)(v6 + 8);
  v24 = *(__int64 **)(v21 + 200);
  v25 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 352LL);
  if ( v25 != sub_2EBDF80 )
    v95 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v25)(v23, v9, *(_QWORD *)(v6 + 768));
  v26 = (_DWORD *)(*(_QWORD *)(v6 + 48) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v95 + 24LL));
  if ( *(_DWORD *)(v6 + 56) != *v26 )
    sub_2F60630(v91, v95, 3LL * *(unsigned __int16 *)(*(_QWORD *)v95 + 24LL), v20);
  v27 = v26[1];
  v28 = v24;
  v97 = v27;
  v100 = &v24[v22];
  do
  {
    v29 = *v28;
    v30 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = *(_QWORD *)(v30 + 16);
    if ( !v31 )
    {
LABEL_69:
      sub_2FB2500(*(_QWORD *)(v6 + 1000));
      v69 = sub_2FBA5C0(*(_QWORD *)(v6 + 1000), v29);
      v70 = sub_2FBA740(*(_QWORD *)(v6 + 1000), v29);
      sub_2FBD930(*(_QWORD *)(v6 + 1000), v69, v70);
      goto LABEL_41;
    }
    if ( *(_WORD *)(v31 + 68) == 20 )
    {
      v74 = *(unsigned int **)(v31 + 32);
      v75 = v74 + 10;
    }
    else
    {
      v32 = *(_QWORD *)(v6 + 776);
      v33 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v32 + 520LL);
      if ( v33 == sub_2DCA430 )
        goto LABEL_21;
      ((void (__fastcall *)(unsigned int **, __int64, _QWORD))v33)(&v105, v32, *(_QWORD *)(v30 + 16));
      v74 = v105;
      v75 = (_DWORD *)v106;
      if ( !v107[0] )
        goto LABEL_21;
    }
    if ( (*v74 & 0xFFF00) == 0 && (*v75 & 0xFFF00) == 0 )
      goto LABEL_41;
LABEL_21:
    if ( v98 )
    {
      v66 = sub_2E8A4A0(v31, *(_DWORD *)(a2 + 112), v95, *(_QWORD *)(v6 + 776), *(__int64 **)(v6 + 8), 1);
      v67 = 0;
      if ( v66 )
      {
        v68 = (_DWORD *)(*(_QWORD *)(v6 + 48) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v66 + 24LL));
        if ( *(_DWORD *)(v6 + 56) != *v68 )
          sub_2F60630(v91, v66, 3LL * *(unsigned __int16 *)(*(_QWORD *)v66 + 24LL), v65);
        v67 = v68[1];
      }
      if ( v97 != v67 )
        goto LABEL_69;
      goto LABEL_41;
    }
    if ( !*(_QWORD *)(a2 + 104) )
      goto LABEL_69;
    v90 = *(_QWORD **)(v6 + 8);
    v86 = *(_QWORD *)(v6 + 16);
    if ( *(_WORD *)(v31 + 68) == 20 )
    {
      v83 = *(_DWORD **)(v31 + 32);
      v84 = v83 + 10;
LABEL_99:
      if ( (*(_BYTE *)(v31 + 44) & 0xC) == 0 && ((*v83 >> 8) & 0xFFF) == ((*v84 >> 8) & 0xFFF) )
        goto LABEL_41;
      goto LABEL_25;
    }
    v34 = *(_QWORD *)(v6 + 776);
    v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v34 + 520LL);
    if ( v35 != sub_2DCA430 )
    {
      ((void (__fastcall *)(_QWORD *, __int64, unsigned __int64))v35)(v103, v34, v31);
      v83 = (_DWORD *)v103[0];
      v84 = (_DWORD *)v103[1];
      if ( v104 )
        goto LABEL_99;
    }
LABEL_25:
    v36 = *(_DWORD *)(a2 + 112);
    v105 = (unsigned int *)v107;
    v106 = 0x800000000LL;
    v88 = v36;
    sub_2E923D0(v31, v36, (__int64)&v105);
    v37 = v105;
    v38 = &v105[4 * (unsigned int)v106];
    if ( v105 != v38 )
    {
      v92 = 0;
      v39 = v105;
      v40 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v41 = *(_QWORD *)(*(_QWORD *)v39 + 32LL) + 40LL * v39[2];
          v42 = *(_BYTE *)(v41 + 3) & 0x10;
          v43 = (*(_DWORD *)v41 >> 8) & 0xFFF;
          if ( !v43 )
            break;
          v71 = (__int64 *)(v90[34] + 16LL * v43);
          v72 = *v71;
          v73 = v71[1];
          if ( v42 )
          {
            if ( (*(_BYTE *)(v41 + 4) & 1) == 0 )
              goto LABEL_81;
LABEL_72:
            v39 += 4;
            if ( v38 == v39 )
              goto LABEL_73;
          }
          else
          {
            v39 += 4;
            v92 |= v72;
            v40 |= v73;
            if ( v38 == v39 )
            {
LABEL_73:
              v89 = v40;
              goto LABEL_31;
            }
          }
        }
        if ( !v42 )
        {
          if ( (*(_BYTE *)(v41 + 4) & 1) == 0 )
          {
            v44 = sub_2EBF1E0(v86, v88);
            v37 = v105;
            v92 = v44;
            v89 = v45;
            goto LABEL_31;
          }
          goto LABEL_72;
        }
        v76 = (__int64 *)v90[34];
        v72 = *v76;
        v73 = v76[1];
        if ( (*(_BYTE *)(v41 + 4) & 1) != 0 )
          goto LABEL_72;
LABEL_81:
        v39 += 4;
        v92 |= ~v72;
        v40 |= ~v73;
        if ( v38 == v39 )
          goto LABEL_73;
      }
    }
    v89 = 0;
    v92 = 0;
LABEL_31:
    if ( v37 != (unsigned int *)v107 )
      _libc_free((unsigned __int64)v37);
    if ( *(_QWORD *)(a2 + 104) )
    {
      v93 = 0;
      v87 = v28;
      v46 = 0;
      v85 = v6;
      v47 = *(_QWORD *)(a2 + 104);
      do
      {
        v48 = (__int64 *)sub_2E09D00((__int64 *)v47, v29);
        if ( v48 != (__int64 *)(*(_QWORD *)v47 + 24LL * *(unsigned int *)(v47 + 8))
          && (*(_DWORD *)((*v48 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v48 >> 1) & 3) <= ((unsigned int)(v29 >> 1)
                                                                                                 & 3
                                                                                                 | *(_DWORD *)(v30 + 24)) )
        {
          v93 |= *(_QWORD *)(v47 + 120);
          v46 |= *(_QWORD *)(v47 + 112);
        }
        v47 = *(_QWORD *)(v47 + 104);
      }
      while ( v47 );
      v49 = v46;
      v6 = v85;
      v28 = v87;
    }
    else
    {
      v93 = 0;
      v49 = 0;
    }
    if ( v92 & ~(v90[37] & v49) | v89 & ~(v90[38] & v93) )
      goto LABEL_69;
LABEL_41:
    ++v28;
  }
  while ( v100 != v28 );
  if ( *(_DWORD *)(v109 + 8) != v115 )
  {
    v50 = *(_QWORD *)(v6 + 1000);
    v105 = (unsigned int *)v107;
    v106 = 0x800000000LL;
    sub_2FBB760(v50, &v105);
    sub_2E01430(
      *(__int64 **)(v6 + 840),
      *(_DWORD *)(a2 + 112),
      (unsigned int *)(*(_QWORD *)v109 + 4LL * v115),
      *(unsigned int *)(v109 + 8) - (unsigned __int64)v115);
    v51 = (unsigned int *)v107;
    v52 = *(_QWORD *)v109 + 4LL * *(unsigned int *)(v109 + 8);
    for ( i = (_DWORD *)(*(_QWORD *)v109 + 4LL * v115); (_DWORD *)v52 != i; ++i )
    {
      v55 = *(unsigned int *)(v6 + 928);
      v56 = *i & 0x7FFFFFFF;
      v57 = v56 + 1;
      if ( (int)v56 + 1 > (unsigned int)v55 && v57 != v55 )
      {
        if ( v57 >= v55 )
        {
          v78 = *(unsigned int *)(v6 + 936);
          v79 = *(unsigned int *)(v6 + 940);
          v80 = v57 - v55;
          if ( v57 > (unsigned __int64)*(unsigned int *)(v6 + 932) )
          {
            v94 = v51;
            v96 = v57 - v55;
            v99 = *(_DWORD *)(v6 + 940);
            v101 = *(_DWORD *)(v6 + 936);
            sub_C8D5F0(v6 + 920, (const void *)(v6 + 936), v57, 8u, v78, v79);
            v55 = *(unsigned int *)(v6 + 928);
            v51 = v94;
            v80 = v96;
            LODWORD(v79) = v99;
            LODWORD(v78) = v101;
          }
          v81 = (_DWORD *)(*(_QWORD *)(v6 + 920) + 8 * v55);
          v82 = v80;
          do
          {
            if ( v81 )
            {
              *v81 = v78;
              v81[1] = v79;
            }
            v81 += 2;
            --v82;
          }
          while ( v82 );
          *(_DWORD *)(v6 + 928) += v80;
        }
        else
        {
          *(_DWORD *)(v6 + 928) = v57;
        }
      }
      v54 = (_DWORD *)(*(_QWORD *)(v6 + 920) + 8 * v56);
      if ( !*v54 )
        *v54 = 4;
    }
    if ( v105 != v51 )
      _libc_free((unsigned __int64)v105);
  }
LABEL_54:
  v58 = v110;
  v59 = *(_BYTE *)(v110 + 36) == 0;
  v108[0] = &unk_4A388F0;
  if ( !v59 )
  {
    v60 = *(_QWORD ***)(v110 + 16);
    v61 = &v60[*(unsigned int *)(v110 + 28)];
    v62 = v60;
    if ( v60 != v61 )
    {
      while ( *v62 != v108 )
      {
        if ( v61 == ++v62 )
          goto LABEL_60;
      }
      v63 = (unsigned int)(*(_DWORD *)(v110 + 28) - 1);
      *(_DWORD *)(v110 + 28) = v63;
      *v62 = v60[v63];
      ++*(_QWORD *)(v58 + 8);
    }
LABEL_60:
    if ( v128 )
      goto LABEL_61;
LABEL_89:
    _libc_free((unsigned __int64)v125);
    goto LABEL_61;
  }
  v77 = sub_C8CA60(v110 + 8, (__int64)v108);
  if ( !v77 )
    goto LABEL_60;
  *v77 = -2;
  ++*(_DWORD *)(v58 + 32);
  ++*(_QWORD *)(v58 + 8);
  if ( !v128 )
    goto LABEL_89;
LABEL_61:
  if ( !v122 )
    _libc_free((unsigned __int64)v119);
  return 0;
}
