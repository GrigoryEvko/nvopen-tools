// Function: sub_300E870
// Address: 0x300e870
//
__int64 __fastcall sub_300E870(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 (*v5)(); // rcx
  __int64 v6; // rdx
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  unsigned __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  int v36; // ebx
  __int64 v37; // r15
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned int v41; // edx
  __int64 v42; // rcx
  __int64 *v43; // r14
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // rbx
  __int64 j; // r12
  __int64 v48; // rdx
  unsigned __int64 v49; // rcx
  __m128i *v50; // r8
  __int64 v51; // r9
  __int64 v53; // r12
  __int64 v54; // r14
  __int64 v55; // r15
  unsigned __int64 v56; // rdx
  __m128i *v57; // r10
  __int64 v58; // rax
  unsigned int v59; // r13d
  __int64 *v60; // rbx
  _QWORD *v61; // rax
  __int64 *v62; // rcx
  __int64 v63; // rsi
  __int64 v64; // r11
  _QWORD *v65; // rdi
  __int64 v66; // r15
  __int64 v67; // r11
  __int64 *v68; // r12
  _QWORD *v69; // rax
  __int64 v70; // r15
  unsigned __int64 v71; // r14
  __int64 v72; // rdi
  __int64 v73; // r10
  __int64 v74; // r11
  __int64 *i; // rax
  __int64 v76; // rax
  __int64 v77; // r9
  unsigned __int64 v78; // r10
  __int64 *v79; // rdx
  __int64 *v80; // rsi
  _QWORD *v81; // rsi
  __int64 v82; // rdi
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r11
  __int64 *v88; // r12
  __int64 *v89; // r15
  __int64 v90; // rax
  __int32 v91; // ebx
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rax
  int v95; // [rsp+4h] [rbp-CCh]
  __m128i *v96; // [rsp+10h] [rbp-C0h]
  int v97; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v98; // [rsp+10h] [rbp-C0h]
  __int64 v99; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v100; // [rsp+18h] [rbp-B8h]
  __int64 v101; // [rsp+18h] [rbp-B8h]
  __int64 v102; // [rsp+18h] [rbp-B8h]
  _QWORD *v103; // [rsp+20h] [rbp-B0h]
  int v104; // [rsp+20h] [rbp-B0h]
  unsigned int v105; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v106; // [rsp+20h] [rbp-B0h]
  unsigned int v107; // [rsp+28h] [rbp-A8h]
  int v108; // [rsp+2Ch] [rbp-A4h]
  __m128i v109; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v110; // [rsp+40h] [rbp-90h]
  __m128i v111; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v112[14]; // [rsp+60h] [rbp-70h] BYREF

  v2 = a1;
  *(_QWORD *)(a1 + 200) = a2;
  *(_QWORD *)(a1 + 208) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v3 = *(_QWORD *)(a1 + 200);
  v4 = *(_QWORD *)(v3 + 16);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v5)(v4, a2, 0);
    v3 = *(_QWORD *)(v2 + 200);
  }
  *(_QWORD *)(v2 + 216) = v6;
  v7 = *(__int64 **)(v2 + 8);
  *(_QWORD *)(v2 + 224) = *(_QWORD *)(v3 + 32);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_108:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_5025C1C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_108;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_5025C1C);
  v11 = *(__int64 **)(v2 + 8);
  *(_QWORD *)(v2 + 232) = v10 + 200;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_104:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_501EACC )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_104;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_501EACC);
  v15 = *(__int64 **)(v2 + 8);
  *(_QWORD *)(v2 + 240) = v14 + 200;
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
LABEL_105:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_501EAFC )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_105;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_501EAFC);
  v19 = *(__int64 **)(v2 + 8);
  *(_QWORD *)(v2 + 248) = v18 + 200;
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_106:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_502A66C )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_106;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_502A66C);
  v23 = *(__int64 **)(v2 + 8);
  *(_QWORD *)(v2 + 256) = v22 + 200;
  v24 = *v23;
  v25 = v23[1];
  if ( v24 == v25 )
LABEL_107:
    BUG();
  while ( *(_UNKNOWN **)v24 != &unk_501E91C )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_107;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_501E91C);
  v27 = *(_QWORD *)(v2 + 256);
  v28 = *(_QWORD *)(v2 + 240);
  *(_QWORD *)(v2 + 264) = *(_QWORD *)(v26 + 200);
  sub_2E12D10(v28, v27, v29, v30, v31);
  v35 = *(_QWORD *)(v2 + 224);
  v108 = *(_DWORD *)(v35 + 64);
  if ( v108 )
  {
    v36 = 0;
    while ( 1 )
    {
      v32 = v36 & 0x7FFFFFFF;
      v38 = v32;
      v39 = *(_QWORD *)(*(_QWORD *)(v35 + 56) + 16 * v32 + 8);
      if ( !v39 )
        goto LABEL_28;
      if ( (*(_BYTE *)(v39 + 4) & 8) == 0 )
        break;
      while ( 1 )
      {
        v39 = *(_QWORD *)(v39 + 32);
        if ( !v39 )
          break;
        if ( (*(_BYTE *)(v39 + 4) & 8) == 0 )
          goto LABEL_32;
      }
      if ( v108 == ++v36 )
        goto LABEL_39;
LABEL_29:
      v35 = *(_QWORD *)(v2 + 224);
    }
LABEL_32:
    v33 = *(_QWORD *)(v2 + 240);
    v40 = *(unsigned int *)(v33 + 160);
    if ( (unsigned int)v40 > (unsigned int)v32 )
    {
      v35 = *(_QWORD *)(v33 + 152);
      v37 = *(_QWORD *)(v35 + 8 * v32);
      if ( v37 )
      {
LABEL_26:
        if ( *(_DWORD *)(v37 + 8) )
        {
          v27 = v37;
          if ( !sub_2E13500(*(_QWORD *)(v2 + 240), v37) )
          {
            v33 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v2 + 256) + 32LL) + 4 * v38);
            if ( (_DWORD)v33 )
            {
              v53 = *(_QWORD *)(v37 + 104);
              if ( v53 )
              {
                v104 = v36;
                v54 = 0;
                v55 = 0;
                v56 = 4;
                v57 = &v111;
                v111.m128i_i64[0] = (__int64)v112;
                v111.m128i_i64[1] = 0x400000000LL;
                v58 = 0;
                v99 = v2;
                v59 = v33;
                while ( 1 )
                {
                  v60 = *(__int64 **)v53;
                  if ( v58 + 1 > v56 )
                  {
                    v96 = v57;
                    sub_C8D5F0((__int64)v57, v112, v58 + 1, 0x10u, v33, v34);
                    v58 = v111.m128i_u32[2];
                    v57 = v96;
                  }
                  v61 = (_QWORD *)(v111.m128i_i64[0] + 16 * v58);
                  *v61 = v53;
                  v61[1] = v60;
                  v58 = (unsigned int)++v111.m128i_i32[2];
                  v62 = *(__int64 **)v53;
                  if ( (v55 & 0xFFFFFFFFFFFFFFF8LL) == 0
                    || (*(_DWORD *)((*v62 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v62 >> 1) & 3) < (*(_DWORD *)((v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v55 >> 1) & 3) )
                  {
                    v55 = *v62;
                  }
                  v63 = *(unsigned int *)(v53 + 8);
                  if ( (v54 & 0xFFFFFFFFFFFFFFF8LL) == 0
                    || (*(_DWORD *)((v62[3 * v63 - 2] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                      | (unsigned int)(v62[3 * v63 - 2] >> 1) & 3) > ((unsigned int)(v54 >> 1) & 3
                                                                    | *(_DWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
                  {
                    v54 = v62[3 * v63 - 2];
                  }
                  v53 = *(_QWORD *)(v53 + 104);
                  if ( !v53 )
                    break;
                  v56 = v111.m128i_u32[3];
                }
                v107 = v59;
                v2 = v99;
                v36 = v104;
                v64 = *(_QWORD *)(v99 + 232);
                v65 = *(_QWORD **)(v64 + 296);
                v109.m128i_i64[0] = v55;
                v66 = (__int64)&v65[2 * *(unsigned int *)(v64 + 304)];
                v27 = v66;
                v68 = sub_300B290(v65, v66, v109.m128i_i64);
                v69 = (_QWORD *)v111.m128i_i64[0];
                if ( (__int64 *)v66 != v68 )
                {
                  v95 = v104;
                  v70 = v111.m128i_i64[0];
                  v33 = v107;
                  v100 = v54 & 0xFFFFFFFFFFFFFFF8LL;
                  v105 = (v54 >> 1) & 3;
                  do
                  {
                    v71 = *v68 & 0xFFFFFFFFFFFFFFF8LL;
                    v34 = (*v68 >> 1) & 3;
                    v32 = (unsigned int)v34 | *(_DWORD *)(v71 + 24);
                    if ( (unsigned int)v32 > (*(_DWORD *)(v100 + 24) | v105) )
                      break;
                    v72 = v70 + 16LL * v111.m128i_u32[2];
                    if ( v72 != v70 )
                    {
                      v35 = v70;
                      v73 = 0;
                      v74 = 0;
                      while ( 2 )
                      {
                        while ( 2 )
                        {
                          v32 = *(_QWORD *)v35;
                          for ( i = *(__int64 **)(v35 + 8); ; *(_QWORD *)(v35 + 8) = i )
                          {
                            v33 = 3LL * *(unsigned int *)(v32 + 8);
                            v27 = *(_QWORD *)v32 + 24LL * *(unsigned int *)(v32 + 8);
                            if ( i == (__int64 *)v27 )
                              goto LABEL_66;
                            v27 = (unsigned int)v34 | *(_DWORD *)(v71 + 24);
                            if ( (*(_DWORD *)((i[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(i[1] >> 1) & 3) > (unsigned int)v27 )
                              break;
                            i += 3;
                          }
                          v33 = *i & 0xFFFFFFFFFFFFFFF8LL;
                          if ( (unsigned int)v27 < (*(_DWORD *)(v33 + 24) | (unsigned int)(*i >> 1) & 3) )
                          {
LABEL_66:
                            v35 += 16;
                            if ( v72 != v35 )
                              continue;
                            goto LABEL_67;
                          }
                          break;
                        }
                        v35 += 16;
                        v74 |= *(_QWORD *)(v32 + 112);
                        v73 |= *(_QWORD *)(v32 + 120);
                        if ( v72 != v35 )
                          continue;
                        break;
                      }
LABEL_67:
                      if ( v74 || v73 )
                      {
                        v76 = v68[1];
                        v27 = (__int64)&v109;
                        v109.m128i_i64[1] = v74;
                        v110 = v73;
                        v109.m128i_i32[0] = (unsigned __int16)v107;
                        sub_300C850((unsigned __int64 *)(v76 + 184), &v109);
                        v70 = v111.m128i_i64[0];
                      }
                      v67 = *(_QWORD *)(v2 + 232);
                    }
                    v68 += 2;
                  }
                  while ( v68 != (__int64 *)(*(_QWORD *)(v67 + 296) + 16LL * *(unsigned int *)(v67 + 304)) );
                  v36 = v95;
                  v69 = (_QWORD *)v70;
                }
                if ( v69 != v112 )
                  _libc_free((unsigned __int64)v69);
              }
              else
              {
                v35 = *(_QWORD *)v37;
                v87 = *(_QWORD *)(v2 + 232);
                v88 = *(__int64 **)(v87 + 296);
                v101 = *(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8);
                if ( *(_QWORD *)v37 != v101 )
                {
                  v97 = v36;
                  v89 = *(__int64 **)v37;
                  v90 = *(_QWORD *)(v87 + 296);
                  v91 = (unsigned __int16)v33;
                  while ( 1 )
                  {
                    v111.m128i_i64[0] = *v89;
                    v27 = v90 + 16LL * *(unsigned int *)(v87 + 304);
                    v88 = sub_300B290(v88, v27, v111.m128i_i64);
                    if ( (__int64 *)v27 != v88 )
                    {
                      do
                      {
                        v32 = *(_DWORD *)((*v88 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v88 >> 1) & 3;
                        v93 = v89[1];
                        v94 = v93 >> 1;
                        v27 = v93 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (unsigned int)v32 >= (*(_DWORD *)(v27 + 24) | (unsigned int)(v94 & 3)) )
                          break;
                        v92 = v88[1];
                        v27 = (__int64)&v111;
                        v111.m128i_i32[0] = v91;
                        v88 += 2;
                        v111.m128i_i64[1] = -1;
                        v112[0] = -1;
                        sub_300C850((unsigned __int64 *)(v92 + 184), &v111);
                        v87 = *(_QWORD *)(v2 + 232);
                      }
                      while ( v88 != (__int64 *)(*(_QWORD *)(v87 + 296) + 16LL * *(unsigned int *)(v87 + 304)) );
                    }
                    v89 += 3;
                    if ( (__int64 *)v101 == v89 )
                      break;
                    v90 = *(_QWORD *)(v87 + 296);
                  }
                  v36 = v97;
                }
              }
            }
          }
        }
LABEL_28:
        if ( v108 == ++v36 )
          goto LABEL_39;
        goto LABEL_29;
      }
    }
    v41 = v32 + 1;
    if ( (unsigned int)v40 < v41 && v41 != v40 )
    {
      if ( v41 >= v40 )
      {
        v77 = *(_QWORD *)(v33 + 168);
        v78 = v41 - v40;
        if ( v41 > (unsigned __int64)*(unsigned int *)(v33 + 164) )
        {
          v98 = v41 - v40;
          v102 = *(_QWORD *)(v33 + 168);
          v106 = *(_QWORD *)(v2 + 240);
          sub_C8D5F0(v33 + 152, (const void *)(v33 + 168), v41, 8u, v33, v77);
          v33 = v106;
          v78 = v98;
          v77 = v102;
          v40 = *(unsigned int *)(v106 + 160);
        }
        v42 = *(_QWORD *)(v33 + 152);
        v79 = (__int64 *)(v42 + 8 * v40);
        v80 = &v79[v78];
        if ( v79 != v80 )
        {
          do
            *v79++ = v77;
          while ( v80 != v79 );
          LODWORD(v40) = *(_DWORD *)(v33 + 160);
          v42 = *(_QWORD *)(v33 + 152);
        }
        *(_DWORD *)(v33 + 160) = v78 + v40;
        goto LABEL_35;
      }
      *(_DWORD *)(v33 + 160) = v41;
    }
    v42 = *(_QWORD *)(v33 + 152);
LABEL_35:
    v103 = (_QWORD *)v33;
    v43 = (__int64 *)(v42 + 8 * v38);
    v44 = sub_2E10F30(v36 | 0x80000000);
    *v43 = v44;
    v27 = v44;
    v37 = v44;
    sub_2E11E80(v103, v44);
    goto LABEL_26;
  }
LABEL_39:
  v45 = *(_QWORD *)(v2 + 200);
  v46 = *(_QWORD *)(v45 + 328);
  for ( j = v45 + 320; j != v46; v46 = *(_QWORD *)(v46 + 8) )
    sub_2E31EE0(v46, v27, v32, v35, v33, v34);
  sub_300C890(v2, v27, v32, v35, v33, v34);
  if ( *(_BYTE *)(v2 + 304) )
  {
    v81 = *(_QWORD **)(v2 + 256);
    sub_2E092B0(*(__int64 **)(v2 + 264), v81, v48, v49, v50, v51);
    v82 = *(_QWORD *)(v2 + 256);
    *(_DWORD *)(v82 + 40) = 0;
    sub_300BAC0(v82, (__int64)v81, v83, v84, v85, v86);
    sub_2EBEAA0(*(_QWORD *)(v2 + 224));
  }
  return 1;
}
