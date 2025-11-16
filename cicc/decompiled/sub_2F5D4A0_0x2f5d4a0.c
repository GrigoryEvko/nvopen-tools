// Function: sub_2F5D4A0
// Address: 0x2f5d4a0
//
__int64 __fastcall sub_2F5D4A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // ecx
  unsigned int v17; // r12d
  char v19; // al
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  char v22; // r14
  size_t v23; // rax
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 **v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 **v33; // rax
  __int64 (__fastcall *v34)(__int64); // rax
  __int64 v35; // rdi
  _DWORD *v36; // rax
  const void *v37; // rsi
  __int64 i; // r13
  _DWORD *v39; // rdx
  unsigned __int64 v40; // rcx
  __int64 v41; // rbx
  unsigned int v42; // edx
  __int64 v43; // rdi
  __int64 (*v44)(); // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  unsigned int v47; // ecx
  __int64 v48; // rax
  int v49; // ebx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // edx
  __int64 v54; // r9
  __int64 v55; // r8
  unsigned __int64 v56; // r10
  _DWORD *v57; // rdx
  unsigned __int64 v58; // rcx
  __int64 (__fastcall *v59)(__int64); // rax
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rdx
  int *v63; // rbx
  int *v64; // r13
  int v65; // esi
  __int64 (__fastcall *v66)(__int64); // rax
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rdx
  int *v70; // r13
  int *v71; // rbx
  int v72; // esi
  __int64 v73; // rbx
  __int64 **v74; // rsi
  __int64 **v75; // rdx
  __int64 **v76; // rax
  __int64 v77; // rcx
  __int64 v78; // rcx
  __int64 v79; // rbx
  _DWORD *v80; // rdx
  unsigned __int64 v81; // rsi
  unsigned int v82; // eax
  const char *v83; // r13
  __int64 *v84; // rax
  __int64 *v85; // rax
  int v86; // ebx
  int v87; // r14d
  _DWORD *v88; // rdx
  __int64 v89; // rcx
  int v90; // r14d
  int v91; // r10d
  _DWORD *v92; // rax
  __int64 v93; // rdx
  _DWORD *v94; // rbx
  _DWORD *v95; // [rsp+8h] [rbp-198h]
  int v96; // [rsp+14h] [rbp-18Ch]
  unsigned int v97; // [rsp+18h] [rbp-188h]
  int v98; // [rsp+18h] [rbp-188h]
  int v99; // [rsp+18h] [rbp-188h]
  unsigned __int64 v100; // [rsp+18h] [rbp-188h]
  int srca; // [rsp+20h] [rbp-180h]
  void *srcb; // [rsp+20h] [rbp-180h]
  int srcc; // [rsp+20h] [rbp-180h]
  int v106; // [rsp+30h] [rbp-170h]
  unsigned int v107; // [rsp+30h] [rbp-170h]
  __int64 v108; // [rsp+30h] [rbp-170h]
  size_t v109; // [rsp+38h] [rbp-168h]
  unsigned __int8 v110; // [rsp+47h] [rbp-159h] BYREF
  int v111[2]; // [rsp+48h] [rbp-158h] BYREF
  unsigned __int64 v112[3]; // [rsp+50h] [rbp-150h] BYREF
  char v113; // [rsp+68h] [rbp-138h] BYREF
  __int64 v114[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v115; // [rsp+B0h] [rbp-F0h]
  __int64 v116; // [rsp+B8h] [rbp-E8h]
  __int64 v117; // [rsp+C0h] [rbp-E0h]
  __int64 v118; // [rsp+C8h] [rbp-D8h]
  __int64 v119; // [rsp+D0h] [rbp-D0h]
  __int64 v120; // [rsp+D8h] [rbp-C8h]
  unsigned int v121; // [rsp+E0h] [rbp-C0h]
  char v122; // [rsp+E4h] [rbp-BCh]
  __int64 v123; // [rsp+E8h] [rbp-B8h]
  __int64 v124; // [rsp+F0h] [rbp-B0h]
  char *v125; // [rsp+F8h] [rbp-A8h]
  __int64 v126; // [rsp+100h] [rbp-A0h]
  int v127; // [rsp+108h] [rbp-98h]
  char v128; // [rsp+10Ch] [rbp-94h]
  char v129; // [rsp+110h] [rbp-90h] BYREF
  __int64 v130; // [rsp+130h] [rbp-70h]
  char *v131; // [rsp+138h] [rbp-68h]
  __int64 v132; // [rsp+140h] [rbp-60h]
  int v133; // [rsp+148h] [rbp-58h]
  char v134; // [rsp+14Ch] [rbp-54h]
  char v135; // [rsp+150h] [rbp-50h] BYREF

  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(unsigned int *)(a2 + 112);
  v12 = *(_QWORD *)(a1 + 40);
  v110 = -1;
  sub_34B8230(v112, v11, v10, a1 + 48, v12);
  v13 = sub_2F5BAC0(a1, a2, (__int64)v112, a3, a4);
  if ( v13 )
  {
    v16 = v13;
    if ( !*(_QWORD *)(a1 + 28944) )
      goto LABEL_3;
    v97 = v13;
    v19 = sub_2F50F60(*(_QWORD *)(a1 + 968), v13);
    v16 = v97;
    if ( !v19 )
      goto LABEL_3;
    if ( *(_DWORD *)(a3 + 8) )
      goto LABEL_3;
    v16 = sub_2F57F90((_QWORD *)a1, a2, (__int64)v112, v97, &v110, a3);
    if ( v16 || *(_DWORD *)(a3 + 8) )
      goto LABEL_3;
    goto LABEL_13;
  }
  if ( !*(_DWORD *)(a3 + 8) )
  {
LABEL_13:
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) != 2 )
    {
      v98 = *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
      v16 = sub_2F51120(a1, a2, (__int64)v112, a3, v110, a4);
      if ( v16 )
      {
        v50 = *(_QWORD *)(a1 + 16);
        v51 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
        if ( (unsigned int)v51 < *(_DWORD *)(v50 + 248) )
        {
          v52 = *(_QWORD *)(v50 + 240) + 40 * v51;
          if ( *(_DWORD *)(v52 + 16) )
          {
            v53 = **(_DWORD **)(v52 + 8);
            if ( !*(_DWORD *)v52 && v53 && v16 != v53 )
            {
              v106 = v16;
              v114[0] = a2;
              sub_2F5B790(a1 + 28952, v114);
              v16 = v106;
            }
          }
        }
        goto LABEL_3;
      }
      if ( v98 <= 1 )
      {
        v45 = *(unsigned int *)(a1 + 928);
        v46 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
        v47 = v46 + 1;
        if ( (int)v46 + 1 > (unsigned int)v45 )
        {
          v14 = v47;
          if ( v47 != v45 )
          {
            if ( v47 >= v45 )
            {
              v86 = *(_DWORD *)(a1 + 936);
              v87 = *(_DWORD *)(a1 + 940);
              v15 = v47 - v45;
              if ( v47 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
              {
                srcb = (void *)(v47 - v45);
                v107 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
                sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v47, 8u, v47, v15);
                v45 = *(unsigned int *)(a1 + 928);
                v15 = (__int64)srcb;
                v46 = v107;
              }
              v88 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v45);
              v89 = v15;
              do
              {
                if ( v88 )
                {
                  *v88 = v86;
                  v88[1] = v87;
                }
                v88 += 2;
                --v89;
              }
              while ( v89 );
              *(_DWORD *)(a1 + 928) += v15;
            }
            else
            {
              *(_DWORD *)(a1 + 928) = v47;
            }
          }
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v46) = 2;
        goto LABEL_44;
      }
      if ( v98 != 3 )
      {
        if ( v98 > 5 )
        {
LABEL_40:
          v43 = *(_QWORD *)(a1 + 8);
          v44 = *(__int64 (**)())(*(_QWORD *)v43 + 656LL);
          if ( v44 == sub_2F4C080
            || ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v44)(v43, *(_QWORD *)(a1 + 768), a2) )
          {
            v16 = sub_2F5C3C0(a1, a2, (__int64)v112, a3, a4, a5, a6);
          }
          else
          {
            v16 = -1;
          }
          goto LABEL_3;
        }
LABEL_15:
        if ( *(float *)(a2 + 116) != INFINITY )
        {
          if ( (_BYTE)qword_5023EC8
            || (v20 = *(_QWORD *)(a1 + 8), v21 = *(__int64 (**)())(*(_QWORD *)v20 + 664LL), v21 != sub_2F4C090)
            && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v21)(v20, *(_QWORD *)(a1 + 768), a2) )
          {
            v78 = *(_QWORD *)(a1 + 920);
            v79 = 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF);
            v80 = (_DWORD *)(v78 + v79);
            if ( *(int *)(v78 + v79) <= 4 )
            {
              v81 = *(unsigned int *)(a1 + 928);
              v82 = (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 1;
              if ( v82 > (unsigned int)v81 )
              {
                v15 = v82;
                if ( v82 != v81 )
                {
                  if ( v82 >= v81 )
                  {
                    v90 = *(_DWORD *)(a1 + 936);
                    v91 = *(_DWORD *)(a1 + 940);
                    v14 = v82 - v81;
                    if ( v82 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
                    {
                      srcc = *(_DWORD *)(a1 + 940);
                      v108 = v82 - v81;
                      sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v82, 8u, v14, v82);
                      v78 = *(_QWORD *)(a1 + 920);
                      v81 = *(unsigned int *)(a1 + 928);
                      v91 = srcc;
                      v14 = v108;
                    }
                    v92 = (_DWORD *)(v78 + 8 * v81);
                    v93 = v14;
                    do
                    {
                      if ( v92 )
                      {
                        *v92 = v90;
                        v92[1] = v91;
                      }
                      v92 += 2;
                      --v93;
                    }
                    while ( v93 );
                    v94 = (_DWORD *)(*(_QWORD *)(a1 + 920) + v79);
                    *(_DWORD *)(a1 + 928) += v14;
                    v80 = v94;
                  }
                  else
                  {
                    *(_DWORD *)(a1 + 928) = v82;
                  }
                }
              }
              *v80 = 5;
LABEL_44:
              v48 = *(unsigned int *)(a3 + 8);
              v49 = *(_DWORD *)(a2 + 112);
              if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
              {
                sub_C8D5F0(a3, (const void *)(a3 + 16), v48 + 1, 4u, v14, v15);
                v48 = *(unsigned int *)(a3 + 8);
              }
              *(_DWORD *)(*(_QWORD *)a3 + 4 * v48) = v49;
              ++*(_DWORD *)(a3 + 8);
              goto LABEL_8;
            }
          }
          v22 = byte_4F826E9[0];
          v109 = strlen("Register Allocation");
          v23 = strlen("regalloc");
          sub_CA08F0(
            (__int64 *)v111,
            "spill",
            5u,
            (__int64)"Spiller",
            7,
            v22,
            "regalloc",
            v23,
            "Register Allocation",
            v109);
          v25 = *(_QWORD *)(a1 + 768);
          v114[1] = a2;
          v26 = *(_QWORD *)(a1 + 24);
          v115 = a3;
          v27 = *(_QWORD *)(a1 + 32);
          v114[0] = (__int64)&unk_4A388F0;
          v28 = *(_QWORD *)(v25 + 32);
          v117 = v27;
          v116 = v28;
          v118 = v26;
          v29 = *(_QWORD *)(v25 + 16);
          v30 = *(__int64 ***)(*(_QWORD *)v29 + 128LL);
          v31 = 0;
          if ( v30 != (__int64 **)sub_2DAC790 )
          {
            v31 = ((__int64 (__fastcall *)(__int64))v30)(v29);
            v28 = v116;
          }
          v119 = v31;
          v32 = *(_DWORD *)(a3 + 8);
          v120 = a1 + 760;
          v121 = v32;
          v125 = &v129;
          v122 = 0;
          v123 = a1 + 400;
          v124 = 0;
          v126 = 4;
          v127 = 0;
          v128 = 1;
          v130 = 0;
          v131 = &v135;
          v132 = 4;
          v133 = 0;
          v134 = 1;
          if ( !*(_BYTE *)(v28 + 36) )
            goto LABEL_94;
          v33 = *(__int64 ***)(v28 + 16);
          v27 = *(unsigned int *)(v28 + 28);
          v30 = &v33[v27];
          if ( v33 != v30 )
          {
            while ( *v33 != v114 )
            {
              if ( v30 == ++v33 )
                goto LABEL_95;
            }
            goto LABEL_25;
          }
LABEL_95:
          if ( (unsigned int)v27 < *(_DWORD *)(v28 + 24) )
          {
            *(_DWORD *)(v28 + 28) = v27 + 1;
            *v30 = v114;
            ++*(_QWORD *)(v28 + 8);
          }
          else
          {
LABEL_94:
            sub_C8CC70(v28 + 8, (__int64)v114, (__int64)v30, v27, v29, v24);
          }
LABEL_25:
          v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL);
          if ( v34 == sub_2F4C0F0 )
            v35 = *(_QWORD *)(a1 + 872);
          else
            v35 = v34(a1);
          (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v35 + 24LL))(v35, v114);
          v36 = *(_DWORD **)a3;
          v37 = (const void *)(a1 + 936);
          for ( i = *(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8); (_DWORD *)i != v36; ++v36 )
          {
            v40 = *(unsigned int *)(a1 + 928);
            v41 = *v36 & 0x7FFFFFFF;
            v42 = v41 + 1;
            if ( (int)v41 + 1 > (unsigned int)v40 && v42 != v40 )
            {
              if ( v42 >= v40 )
              {
                v54 = *(unsigned int *)(a1 + 936);
                v55 = *(unsigned int *)(a1 + 940);
                v56 = v42 - v40;
                if ( v42 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
                {
                  v95 = v36;
                  v96 = *(_DWORD *)(a1 + 936);
                  v100 = v42 - v40;
                  srca = *(_DWORD *)(a1 + 940);
                  sub_C8D5F0(a1 + 920, v37, v42, 8u, v55, v54);
                  v40 = *(unsigned int *)(a1 + 928);
                  v36 = v95;
                  LODWORD(v54) = v96;
                  v56 = v100;
                  LODWORD(v55) = srca;
                }
                v57 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v40);
                v58 = v56;
                do
                {
                  if ( v57 )
                  {
                    *v57 = v54;
                    v57[1] = v55;
                  }
                  v57 += 2;
                  --v58;
                }
                while ( v58 );
                *(_DWORD *)(a1 + 928) += v56;
              }
              else
              {
                *(_DWORD *)(a1 + 928) = v42;
              }
            }
            v39 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v41);
            if ( !*v39 )
              *v39 = 6;
          }
          v59 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL);
          if ( v59 == sub_2F4C0F0 )
            v60 = *(_QWORD *)(a1 + 872);
          else
            v60 = ((__int64 (__fastcall *)(__int64, const void *))v59)(a1, v37);
          v61 = (*(__int64 (__fastcall **)(__int64, const void *))(*(_QWORD *)v60 + 32LL))(v60, v37);
          v63 = (int *)(v61 + 4 * v62);
          v64 = (int *)v61;
          if ( (int *)v61 != v63 )
          {
            do
            {
              v65 = *v64++;
              sub_2E01430(
                *(__int64 **)(a1 + 840),
                v65,
                (unsigned int *)(*(_QWORD *)v115 + 4LL * v121),
                *(unsigned int *)(v115 + 8) - (unsigned __int64)v121);
            }
            while ( v63 != v64 );
          }
          v66 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL);
          if ( v66 == sub_2F4C0F0 )
            v67 = *(_QWORD *)(a1 + 872);
          else
            v67 = v66(a1);
          v68 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v67 + 40LL))(v67);
          v70 = (int *)(v68 + 4 * v69);
          v71 = (int *)v68;
          if ( (int *)v68 != v70 )
          {
            do
            {
              v72 = *v71++;
              sub_2E01430(
                *(__int64 **)(a1 + 840),
                v72,
                (unsigned int *)(*(_QWORD *)v115 + 4LL * v121),
                *(unsigned int *)(v115 + 8) - (unsigned __int64)v121);
            }
            while ( v70 != v71 );
          }
          if ( unk_503FCFD )
          {
            v83 = *(const char **)(a1 + 768);
            v84 = (__int64 *)sub_CB72A0();
            sub_2F06090(v83, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 784), (__int64)"After spilling", v84, 1);
          }
          v73 = v116;
          v114[0] = (__int64)&unk_4A388F0;
          if ( *(_BYTE *)(v116 + 36) )
          {
            v74 = *(__int64 ***)(v116 + 16);
            v75 = &v74[*(unsigned int *)(v116 + 28)];
            v76 = v74;
            if ( v74 != v75 )
            {
              while ( *v76 != v114 )
              {
                if ( v75 == ++v76 )
                  goto LABEL_79;
              }
              v77 = (unsigned int)(*(_DWORD *)(v116 + 28) - 1);
              *(_DWORD *)(v116 + 28) = v77;
              *v76 = v74[v77];
              ++*(_QWORD *)(v73 + 8);
            }
          }
          else
          {
            v85 = sub_C8CA60(v116 + 8, (__int64)v114);
            if ( v85 )
            {
              *v85 = -2;
              ++*(_DWORD *)(v73 + 32);
              ++*(_QWORD *)(v73 + 8);
            }
          }
LABEL_79:
          if ( !v134 )
            _libc_free((unsigned __int64)v131);
          if ( !v128 )
            _libc_free((unsigned __int64)v125);
          if ( *(_QWORD *)v111 )
            sub_C9E2A0(*(__int64 *)v111);
          goto LABEL_8;
        }
        goto LABEL_40;
      }
    }
    if ( !*(_DWORD *)(a2 + 8) )
      goto LABEL_15;
    v99 = *(_DWORD *)(a3 + 8);
    v16 = sub_2F57C30(a1, a2, (__int64)v112, a3);
    if ( !v16 && v99 == *(_DWORD *)(a3 + 8) )
      goto LABEL_15;
LABEL_3:
    v17 = v16;
    goto LABEL_4;
  }
LABEL_8:
  v17 = 0;
LABEL_4:
  if ( (char *)v112[0] != &v113 )
    _libc_free(v112[0]);
  return v17;
}
