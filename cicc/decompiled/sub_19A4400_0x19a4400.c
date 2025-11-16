// Function: sub_19A4400
// Address: 0x19a4400
//
void __fastcall sub_19A4400(__int64 a1, __m128i a2, __m128i a3)
{
  __int64 v3; // r15
  __int64 v4; // r12
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // r8d
  _QWORD *v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // r13
  int v19; // eax
  __int64 v20; // rax
  int v21; // ecx
  int v22; // r8d
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // r9
  __int64 v26; // r14
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rax
  bool v31; // al
  __int64 v32; // rsi
  __int64 *v33; // rdx
  unsigned __int64 v34; // rbx
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // edx
  __int64 v40; // rdx
  __int64 v41; // rdi
  unsigned int v42; // esi
  __int64 *v43; // rcx
  __int64 v44; // r9
  _QWORD *v45; // rbx
  __int64 v46; // r9
  _QWORD *v47; // r8
  __int64 v48; // r9
  __int64 v49; // r8
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rbx
  __int64 v53; // rsi
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r14
  __int64 *v57; // rbx
  __int64 *v58; // r12
  __int64 v59; // rsi
  __int64 v60; // rsi
  bool v61; // al
  __int64 v62; // rdx
  int v63; // r9d
  __int64 v64; // r8
  __int64 v65; // rcx
  __int64 v66; // rdi
  __int64 v67; // r14
  __int64 v68; // r15
  __int64 v69; // r12
  __int64 v70; // r13
  bool v71; // al
  bool v72; // al
  _QWORD *v73; // rdx
  char v74; // al
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rcx
  unsigned __int64 v78; // rdi
  __int64 v79; // rcx
  __int64 v80; // rdx
  unsigned __int64 v81; // rcx
  __int64 v82; // rdx
  int v83; // ecx
  int v84; // r10d
  __int64 v85; // rdx
  unsigned __int64 v86; // rcx
  __int64 v87; // [rsp+8h] [rbp-198h]
  __int64 v88; // [rsp+10h] [rbp-190h]
  __int64 v89; // [rsp+18h] [rbp-188h]
  __int64 v90; // [rsp+20h] [rbp-180h]
  __int64 v91; // [rsp+20h] [rbp-180h]
  __int64 v92; // [rsp+28h] [rbp-178h]
  __int64 v93; // [rsp+28h] [rbp-178h]
  unsigned int v94; // [rsp+28h] [rbp-178h]
  __int64 *v95; // [rsp+30h] [rbp-170h]
  _QWORD **v96; // [rsp+38h] [rbp-168h]
  __int64 v97; // [rsp+38h] [rbp-168h]
  __int64 *v98; // [rsp+38h] [rbp-168h]
  __int64 v99; // [rsp+40h] [rbp-160h]
  __int64 v100; // [rsp+48h] [rbp-158h]
  __int64 v101; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v102; // [rsp+68h] [rbp-138h] BYREF
  _BYTE *v103; // [rsp+70h] [rbp-130h] BYREF
  __int64 v104; // [rsp+78h] [rbp-128h]
  _BYTE v105[32]; // [rsp+80h] [rbp-120h] BYREF
  _BYTE *v106; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v107; // [rsp+A8h] [rbp-F8h]
  _BYTE v108[32]; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v109; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-C8h]
  unsigned __int64 v111; // [rsp+E0h] [rbp-C0h]
  char v112[24]; // [rsp+F8h] [rbp-A8h] BYREF
  __int128 v113; // [rsp+110h] [rbp-90h] BYREF
  __int128 v114; // [rsp+120h] [rbp-80h] BYREF
  __int128 *v115; // [rsp+130h] [rbp-70h] BYREF
  __int64 v116; // [rsp+138h] [rbp-68h]
  __int128 v117; // [rsp+140h] [rbp-60h] BYREF
  __int128 v118; // [rsp+150h] [rbp-50h]
  __int64 v119; // [rsp+160h] [rbp-40h]
  __int64 v120; // [rsp+168h] [rbp-38h]

  v3 = *(_QWORD *)(*(_QWORD *)a1 + 216LL);
  v99 = *(_QWORD *)a1 + 208LL;
  if ( v3 != v99 )
  {
    v4 = a1;
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      v15 = *(_QWORD *)(v3 - 8);
      *(_QWORD *)&v113 = *(_QWORD *)(v3 + 40);
      v16 = 3LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      {
        v5 = *(_QWORD **)(v15 - 8);
        v6 = (__int64)&v5[v16];
      }
      else
      {
        v6 = v15;
        v5 = (_QWORD *)(v15 - v16 * 8);
      }
      v7 = sub_1992F00(v5, v6, (__int64 *)&v113);
      v10 = *(_QWORD **)(v4 + 32720);
      v11 = (__int64)v7;
      v12 = *(_QWORD **)(v4 + 32712);
      if ( v10 == v12 )
      {
        v13 = &v12[*(unsigned int *)(v4 + 32732)];
        if ( v12 == v13 )
        {
          v73 = *(_QWORD **)(v4 + 32712);
        }
        else
        {
          do
          {
            if ( v11 == *v12 )
              break;
            ++v12;
          }
          while ( v13 != v12 );
          v73 = v13;
        }
        goto LABEL_41;
      }
      v13 = &v10[*(unsigned int *)(v4 + 32728)];
      v12 = sub_16CC9F0(v4 + 32704, v11);
      if ( v11 == *v12 )
        break;
      v14 = *(_QWORD *)(v4 + 32720);
      if ( v14 == *(_QWORD *)(v4 + 32712) )
      {
        v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(v4 + 32732));
        v73 = v12;
LABEL_41:
        while ( v73 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
          ++v12;
        goto LABEL_8;
      }
      v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(v4 + 32728));
LABEL_8:
      if ( v13 == v12 )
      {
        v17 = 0;
        v96 = 0;
        v18 = 0xFFFFFFFFLL;
        if ( (unsigned __int8)sub_1994130(*(_QWORD *)(v4 + 32), v15, *(_QWORD *)(v3 + 40), v8, v9) )
        {
          v17 = 2;
          v38 = sub_19927B0(*(_QWORD *)(v4 + 32), (__int64 *)v15, *(__int64 **)(v3 + 40));
          v18 = v39;
          v96 = (_QWORD **)v38;
        }
        v101 = sub_13CA510(*(_QWORD *)v4, v3 - 32);
        sub_16CCCB0(&v109, (__int64)v112, v3 + 48);
        if ( *(_WORD *)(v101 + 24) != 10 )
        {
          if ( dword_4FB1500 )
          {
            v40 = *(unsigned int *)(v4 + 32872);
            if ( (_DWORD)v40 )
            {
              v41 = *(_QWORD *)(v4 + 32856);
              v42 = (v40 - 1) & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
              v43 = (__int64 *)(v41 + 16LL * v42);
              v44 = *v43;
              if ( v101 == *v43 )
              {
LABEL_48:
                if ( v43 != (__int64 *)(v41 + 16 * v40) )
                  v101 = v43[1];
              }
              else
              {
                v83 = 1;
                while ( v44 != -8 )
                {
                  v84 = v83 + 1;
                  v42 = (v40 - 1) & (v83 + v42);
                  v43 = (__int64 *)(v41 + 16LL * v42);
                  v44 = *v43;
                  if ( v101 == *v43 )
                    goto LABEL_48;
                  v83 = v84;
                }
              }
            }
          }
          if ( *(_BYTE *)(v15 + 16) != 75
            || (v19 = *(unsigned __int16 *)(v15 + 18), BYTE1(v19) &= ~0x80u, (unsigned int)(v19 - 32) > 1) )
          {
LABEL_19:
            v20 = sub_199DBD0(v4, &v101, v17, v96, v18, a2, a3);
            a2 = 0;
            v116 = 2;
            v23 = v20;
            v114 = 0;
            v97 = v24;
            v113 = 0;
            v25 = *(_QWORD *)(v4 + 368) + 1984 * v20;
            *((_QWORD *)&v114 + 1) = (char *)&v117 + 8;
            v115 = (__int128 *)((char *)&v117 + 8);
            v26 = v25;
            v117 = 0;
            v118 = 0;
            v27 = *(_DWORD *)(v25 + 64);
            if ( v27 >= *(_DWORD *)(v25 + 68) )
            {
              sub_19957D0((unsigned __int64 *)(v25 + 56), 0);
              v27 = *(_DWORD *)(v26 + 64);
            }
            v28 = *(_QWORD *)(v26 + 56) + 80LL * v27;
            if ( v28 )
            {
              v92 = *(_QWORD *)(v26 + 56) + 80LL * v27;
              *(_OWORD *)v28 = v113;
              sub_16CCEE0((_QWORD *)(v28 + 16), v28 + 56, 2, (__int64)&v114);
              *(_QWORD *)(v92 + 72) = *((_QWORD *)&v118 + 1);
              v27 = *(_DWORD *)(v26 + 64);
            }
            *(_DWORD *)(v26 + 64) = v27 + 1;
            if ( v115 != *((__int128 **)&v114 + 1) )
              _libc_free((unsigned __int64)v115);
            v29 = (__int64 *)(*(_QWORD *)(v26 + 56) + 80LL * *(unsigned int *)(v26 + 64) - 80);
            *v29 = v15;
            v29[1] = *(_QWORD *)(v3 + 40);
            if ( v29 + 2 != &v109 )
            {
              v95 = v29;
              sub_16CCD50((__int64)(v29 + 2), (__int64)&v109, (__int64)v29, v21, v22, v25);
              v29 = v95;
            }
            v30 = v97;
            v98 = v29;
            v29[9] = v30;
            v31 = sub_19A2CE0(v29, *(_QWORD *)(v4 + 40));
            v32 = *(_QWORD *)(v26 + 736);
            *(_BYTE *)(v26 + 728) &= v31;
            v33 = v98;
            if ( !v32
              || (v34 = sub_1456C90(*(_QWORD *)(v4 + 8), v32),
                  v35 = sub_1456C90(*(_QWORD *)(v4 + 8), *(_QWORD *)v98[1]),
                  v33 = v98,
                  v34 < v35) )
            {
              *(_QWORD *)(v26 + 736) = *(_QWORD *)v33[1];
            }
            if ( !*(_DWORD *)(v26 + 752) )
            {
              v45 = (_QWORD *)v101;
              if ( !(unsigned __int8)sub_3870AF0(v101, *(_QWORD *)(v4 + 8)) )
                *(_BYTE *)(v26 + 729) = 1;
              v46 = *(_QWORD *)(v4 + 40);
              v47 = *(_QWORD **)(v4 + 8);
              v103 = v105;
              v115 = &v117;
              v106 = v108;
              v87 = (__int64)v47;
              v90 = v46;
              v113 = 0u;
              LOBYTE(v114) = 0;
              *((_QWORD *)&v114 + 1) = 0;
              v116 = 0x400000000LL;
              v119 = 0;
              v120 = 0;
              v104 = 0x400000000LL;
              v107 = 0x400000000LL;
              sub_199D530(v45, v46, (__int64)&v103, (__int64)&v106, v47, (__m128i)0LL, a3);
              v48 = v90;
              v49 = v87;
              if ( (_DWORD)v104 )
              {
                v102 = sub_147DD40(v87, (__int64 *)&v103, 0, 0, (__m128i)0LL, a3);
                v72 = sub_14560B0((__int64)v102);
                v49 = v87;
                v48 = v90;
                if ( !v72 )
                {
                  sub_1458920((__int64)&v115, &v102);
                  v48 = v90;
                  v49 = v87;
                }
                LOBYTE(v114) = 1;
              }
              if ( (_DWORD)v107 )
              {
                v91 = v48;
                v102 = sub_147DD40(v49, (__int64 *)&v106, 0, 0, (__m128i)0LL, a3);
                v71 = sub_14560B0((__int64)v102);
                v48 = v91;
                if ( !v71 )
                {
                  sub_1458920((__int64)&v115, &v102);
                  v48 = v91;
                }
                LOBYTE(v114) = 1;
              }
              sub_19932F0((__int64)&v113, v48);
              if ( v106 != v108 )
                _libc_free((unsigned __int64)v106);
              if ( v103 != v105 )
                _libc_free((unsigned __int64)v103);
              sub_19A1660(v4, v26, v23, (__int64)&v113, v50, v51);
              if ( v115 != &v117 )
                _libc_free((unsigned __int64)v115);
              v52 = *(_QWORD *)(v26 + 744) + 96LL * *(unsigned int *)(v26 + 752) - 96;
              v53 = *(_QWORD *)(v52 + 80);
              if ( v53 )
                sub_1998430(v4 + 32128, v53, v23);
              v54 = *(__int64 **)(v52 + 32);
              v55 = *(unsigned int *)(v52 + 40);
              if ( v54 != &v54[v55] )
              {
                v100 = v4;
                v56 = v4 + 32128;
                v57 = &v54[v55];
                v58 = v54;
                do
                {
                  v59 = *v58++;
                  sub_1998430(v56, v59, v23);
                }
                while ( v57 != v58 );
                v4 = v100;
              }
            }
            v36 = v111;
            if ( v111 == v110 )
              goto LABEL_9;
            goto LABEL_32;
          }
          v60 = *(_QWORD *)(v15 - 24);
          if ( *(_QWORD *)(v3 + 40) != v60 )
          {
LABEL_70:
            v93 = sub_146F1B0(*(_QWORD *)(v4 + 8), v60);
            v61 = sub_146CEE0(*(_QWORD *)(v4 + 8), v93, *(_QWORD *)(v4 + 40));
            v64 = v93;
            if ( v61 )
            {
              v74 = sub_3870AF0(v93, *(_QWORD *)(v4 + 8));
              v64 = v93;
              if ( v74 )
              {
                v17 = 3;
                v75 = sub_1499950(v93, (__int64)&v109, *(_QWORD **)(v4 + 8), a2, a3);
                v101 = sub_14806B0(*(_QWORD *)(v4 + 8), v75, v101, 0, 0);
              }
            }
            v65 = *(unsigned int *)(v4 + 200);
            v66 = v4 + 64;
            if ( *(_DWORD *)(v4 + 200) )
            {
              v94 = v17;
              v67 = *(unsigned int *)(v4 + 200);
              v89 = v3;
              v68 = v4;
              v69 = v4 + 64;
              v88 = v18;
              v70 = 0;
              do
              {
                v62 = *(_QWORD *)(*(_QWORD *)(v68 + 192) + 8 * v70);
                if ( v62 != -1 )
                {
                  *(_QWORD *)&v113 = -v62;
                  sub_1994C30(v69, (__int64 *)&v113, -v62, v65, v64, v63);
                }
                ++v70;
              }
              while ( v70 != v67 );
              v66 = v69;
              v17 = v94;
              v4 = v68;
              v3 = v89;
              v18 = v88;
            }
            *(_QWORD *)&v113 = -1;
            sub_1994C30(v66, (__int64 *)&v113, v62, v65, v64, v63);
            goto LABEL_19;
          }
          v76 = *(_QWORD *)(v15 - 48);
          if ( v76 )
          {
            if ( v60 )
            {
              v77 = *(_QWORD *)(v15 - 16);
              v78 = *(_QWORD *)(v15 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v78 = v77;
              if ( v77 )
                *(_QWORD *)(v77 + 16) = v78 | *(_QWORD *)(v77 + 16) & 3LL;
            }
            *(_QWORD *)(v15 - 24) = v76;
            v79 = *(_QWORD *)(v76 + 8);
            *(_QWORD *)(v15 - 16) = v79;
            if ( v79 )
              *(_QWORD *)(v79 + 16) = (v15 - 16) | *(_QWORD *)(v79 + 16) & 3LL;
            *(_QWORD *)(v15 - 8) = (v76 + 8) | *(_QWORD *)(v15 - 8) & 3LL;
            *(_QWORD *)(v76 + 8) = v15 - 24;
            goto LABEL_93;
          }
          if ( v60 )
          {
            v85 = *(_QWORD *)(v15 - 16);
            v86 = *(_QWORD *)(v15 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v86 = v85;
            if ( v85 )
              *(_QWORD *)(v85 + 16) = v86 | *(_QWORD *)(v85 + 16) & 3LL;
            *(_QWORD *)(v15 - 24) = 0;
LABEL_93:
            if ( *(_QWORD *)(v15 - 48) )
            {
              v80 = *(_QWORD *)(v15 - 40);
              v81 = *(_QWORD *)(v15 - 32) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v81 = v80;
              if ( v80 )
                *(_QWORD *)(v80 + 16) = v81 | *(_QWORD *)(v80 + 16) & 3LL;
            }
            *(_QWORD *)(v15 - 48) = v60;
            if ( v60 )
            {
              v82 = *(_QWORD *)(v60 + 8);
              *(_QWORD *)(v15 - 40) = v82;
              if ( v82 )
                *(_QWORD *)(v82 + 16) = (v15 - 40) | *(_QWORD *)(v82 + 16) & 3LL;
              *(_QWORD *)(v15 - 32) = (v60 + 8) | *(_QWORD *)(v15 - 32) & 3LL;
              *(_QWORD *)(v60 + 8) = v15 - 48;
            }
            v60 = *(_QWORD *)(v15 - 24);
          }
          *(_BYTE *)(v4 + 48) = 1;
          goto LABEL_70;
        }
        v36 = v111;
        if ( v110 == v111 )
          goto LABEL_9;
LABEL_32:
        _libc_free(v36);
        v3 = *(_QWORD *)(v3 + 8);
        if ( v99 == v3 )
          return;
      }
      else
      {
LABEL_9:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v99 == v3 )
          return;
      }
    }
    v37 = *(_QWORD *)(v4 + 32720);
    if ( v37 == *(_QWORD *)(v4 + 32712) )
      v8 = *(unsigned int *)(v4 + 32732);
    else
      v8 = *(unsigned int *)(v4 + 32728);
    v73 = (_QWORD *)(v37 + 8 * v8);
    goto LABEL_41;
  }
}
