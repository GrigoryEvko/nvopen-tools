// Function: sub_22B90A0
// Address: 0x22b90a0
//
__int64 __fastcall sub_22B90A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // edi
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int *v11; // rax
  unsigned int *v12; // r12
  unsigned int v13; // r14d
  unsigned int *v14; // rbx
  int *v15; // rax
  unsigned int v16; // edx
  int *v17; // rcx
  int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // ecx
  int *v22; // rsi
  int v23; // r8d
  int *v24; // rsi
  int *v25; // rsi
  int *v26; // rax
  unsigned int v27; // esi
  int *v28; // rdi
  int v29; // r8d
  int *v30; // rsi
  __int64 v31; // r9
  __int64 v32; // rdi
  unsigned int v33; // r8d
  int *v34; // rsi
  int v35; // r10d
  __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned int v38; // r9d
  int *v39; // r8
  int v40; // r10d
  int v41; // esi
  int v42; // esi
  int v43; // r10d
  int v44; // edi
  int v45; // r9d
  __int64 *v46; // r12
  __int64 *v47; // rbx
  __int64 v48; // r11
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rcx
  unsigned int v52; // edi
  __int64 *v53; // rdx
  __int64 v54; // r8
  unsigned int v55; // r13d
  __int64 v56; // rdx
  __int64 v57; // r8
  unsigned int v58; // edi
  int v59; // r9d
  __int64 v60; // r14
  __m128i v61; // xmm0
  __m128i v62; // xmm1
  unsigned int v63; // edi
  __int64 *v64; // rsi
  __int64 v65; // r9
  unsigned int v66; // eax
  int *v67; // rcx
  int v68; // esi
  __int64 v69; // rax
  __int64 v70; // rsi
  unsigned int v71; // edx
  int *v72; // rcx
  int v73; // edi
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned int v76; // esi
  int *v77; // rax
  int v78; // edi
  __int64 v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rdi
  unsigned int v82; // ecx
  __int64 *v83; // rax
  __int64 v84; // r8
  __int64 v85; // rax
  __int64 v86; // rsi
  unsigned int v87; // edx
  int *v88; // rcx
  int v89; // edi
  int v90; // r14d
  int v91; // edx
  int v92; // r10d
  int v93; // esi
  int v94; // ecx
  int v95; // r9d
  int v96; // ecx
  int v97; // r9d
  int v98; // eax
  int v99; // r9d
  int v100; // eax
  int v101; // r10d
  int v102; // ecx
  int v103; // r9d
  int v104; // r8d
  int v106; // [rsp+10h] [rbp-1B0h]
  int v107; // [rsp+10h] [rbp-1B0h]
  int v108; // [rsp+20h] [rbp-1A0h]
  int v109; // [rsp+24h] [rbp-19Ch]
  int v110; // [rsp+28h] [rbp-198h]
  int v111; // [rsp+2Ch] [rbp-194h]
  __int64 v112; // [rsp+30h] [rbp-190h]
  int v114; // [rsp+70h] [rbp-150h]
  unsigned int v115; // [rsp+74h] [rbp-14Ch]
  __int64 v116; // [rsp+78h] [rbp-148h]
  __int64 v117; // [rsp+80h] [rbp-140h]
  __int64 v118; // [rsp+80h] [rbp-140h]
  __int64 v119; // [rsp+80h] [rbp-140h]
  int v120; // [rsp+80h] [rbp-140h]
  unsigned int v121; // [rsp+88h] [rbp-138h]
  __int64 v122; // [rsp+90h] [rbp-130h] BYREF
  __int64 v123; // [rsp+98h] [rbp-128h]
  __int64 v124; // [rsp+A0h] [rbp-120h]
  __int64 v125; // [rsp+A8h] [rbp-118h]
  __int64 v126; // [rsp+B0h] [rbp-110h] BYREF
  __int64 *v127; // [rsp+B8h] [rbp-108h]
  __int64 v128; // [rsp+C0h] [rbp-100h]
  __int64 v129; // [rsp+C8h] [rbp-F8h]
  _OWORD v130[2]; // [rsp+D0h] [rbp-F0h] BYREF
  _BYTE v131[16]; // [rsp+F0h] [rbp-D0h] BYREF
  void (__fastcall *v132)(_QWORD, _QWORD, _QWORD); // [rsp+100h] [rbp-C0h]
  __int64 v133; // [rsp+108h] [rbp-B8h]
  __m128i v134; // [rsp+110h] [rbp-B0h] BYREF
  __m128i v135; // [rsp+120h] [rbp-A0h] BYREF
  _BYTE v136[16]; // [rsp+130h] [rbp-90h] BYREF
  void (__fastcall *v137)(_BYTE *, _BYTE *, __int64); // [rsp+140h] [rbp-80h]
  __int64 v138; // [rsp+148h] [rbp-78h]
  _BYTE v139[16]; // [rsp+170h] [rbp-50h] BYREF
  void (__fastcall *v140)(_BYTE *, _BYTE *, __int64); // [rsp+180h] [rbp-40h]

  v5 = *(_DWORD *)(a3 + 16);
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  if ( v5 )
  {
    v11 = *(unsigned int **)(a3 + 8);
    v12 = &v11[10 * *(unsigned int *)(a3 + 24)];
    if ( v11 != v12 )
    {
      while ( 1 )
      {
        v13 = *v11;
        v14 = v11;
        if ( *v11 <= 0xFFFFFFFD )
          break;
        v11 += 10;
        if ( v12 == v11 )
          goto LABEL_2;
      }
      if ( v12 != v11 )
      {
        while ( 1 )
        {
          v15 = (int *)*((_QWORD *)v14 + 2);
          v16 = v14[6];
          v17 = &v15[v14[8]];
          if ( v16 <= 1 )
          {
            if ( v16 )
            {
              v18 = *v15;
              if ( v15 != v17 )
              {
                while ( 1 )
                {
                  v18 = *v15;
                  v24 = v15;
                  if ( (unsigned int)*v15 <= 0xFFFFFFFD )
                    break;
                  if ( v17 == ++v15 )
                  {
                    v18 = v24[1];
                    break;
                  }
                }
              }
            }
            else
            {
              v18 = *v17;
            }
LABEL_13:
            LODWORD(v126) = v18;
            goto LABEL_14;
          }
          v25 = (int *)*((_QWORD *)v14 + 2);
          if ( v15 != v17 )
          {
            while ( 1 )
            {
              v18 = *v25;
              v26 = v25;
              if ( (unsigned int)*v25 <= 0xFFFFFFFD )
                break;
              if ( v17 == ++v25 )
                goto LABEL_14;
            }
            if ( v25 != v17 )
              break;
          }
LABEL_14:
          sub_22B6470((__int64)&v134, (__int64)&v122, (int *)&v126);
          v19 = *(_QWORD *)(a2 + 96);
          v20 = *(unsigned int *)(a2 + 112);
          if ( (_DWORD)v20 )
          {
            v21 = (v20 - 1) & (37 * v126);
            v22 = (int *)(v19 + 8LL * v21);
            v23 = *v22;
            if ( (_DWORD)v126 == *v22 )
            {
LABEL_16:
              if ( v22 != (int *)(v19 + 8 * v20) )
                v121 = v22[1];
            }
            else
            {
              v42 = 1;
              while ( v23 != -1 )
              {
                v43 = v42 + 1;
                v21 = (v20 - 1) & (v42 + v21);
                v22 = (int *)(v19 + 8LL * v21);
                v23 = *v22;
                if ( (_DWORD)v126 == *v22 )
                  goto LABEL_16;
                v42 = v43;
              }
            }
          }
          v14 += 10;
          *(_QWORD *)&v130[0] = __PAIR64__(v13, v121);
          sub_22B89D0((__int64)&v134, a1 + 120, (int *)v130, (int *)v130 + 1);
          *(_QWORD *)&v130[0] = __PAIR64__(v121, v13);
          sub_22B89D0((__int64)&v134, a1 + 88, (int *)v130, (int *)v130 + 1);
          if ( v14 == v12 )
            goto LABEL_2;
          while ( *v14 > 0xFFFFFFFD )
          {
            v14 += 10;
            if ( v12 == v14 )
              goto LABEL_2;
          }
          if ( v12 == v14 )
            goto LABEL_2;
          v13 = *v14;
        }
        v114 = v125 - 1;
        if ( !(_DWORD)v125 )
          goto LABEL_42;
        while ( 1 )
        {
          v27 = v114 & (37 * v18);
          v28 = (int *)(v123 + 4LL * v27);
          v29 = *v28;
          if ( *v28 == v18 )
            break;
          v44 = 1;
          while ( v29 != -1 )
          {
            v27 = v114 & (v44 + v27);
            v45 = v44 + 1;
            v28 = (int *)(v123 + 4LL * v27);
            v29 = *v28;
            if ( *v28 == v18 )
              goto LABEL_35;
            v44 = v45;
          }
          do
          {
LABEL_42:
            v31 = *(_QWORD *)(a4 + 8);
            v32 = *(unsigned int *)(a4 + 24);
            if ( !(_DWORD)v32 )
              goto LABEL_50;
            v33 = (v32 - 1) & (37 * v18);
            v34 = (int *)(v31 + 40LL * v33);
            v35 = *v34;
            if ( *v34 != v18 )
            {
              v41 = 1;
              while ( v35 != -1 )
              {
                v33 = (v32 - 1) & (v41 + v33);
                v107 = v41 + 1;
                v34 = (int *)(v31 + 40LL * v33);
                v35 = *v34;
                if ( *v34 == v18 )
                  goto LABEL_44;
                v41 = v107;
              }
LABEL_50:
              v34 = (int *)(v31 + 40 * v32);
            }
LABEL_44:
            v36 = *((_QWORD *)v34 + 2);
            v37 = (unsigned int)v34[8];
            if ( (_DWORD)v37 )
            {
              v38 = (v37 - 1) & (37 * v13);
              v39 = (int *)(v36 + 4LL * v38);
              v40 = *v39;
              if ( v13 == *v39 )
              {
LABEL_46:
                if ( v39 != (int *)(v36 + 4 * v37) )
                  goto LABEL_13;
              }
              else
              {
                v104 = 1;
                while ( v40 != -1 )
                {
                  v38 = (v37 - 1) & (v104 + v38);
                  v106 = v104 + 1;
                  v39 = (int *)(v36 + 4LL * v38);
                  v40 = *v39;
                  if ( *v39 == v13 )
                    goto LABEL_46;
                  v104 = v106;
                }
              }
            }
LABEL_36:
            v30 = v26 + 1;
            if ( v26 + 1 != v17 )
            {
              while ( 1 )
              {
                v18 = *v30;
                v26 = v30;
                if ( (unsigned int)*v30 <= 0xFFFFFFFD )
                  break;
                if ( v17 == ++v30 )
                  goto LABEL_14;
              }
              if ( v30 != v17 )
                continue;
            }
            goto LABEL_14;
          }
          while ( !(_DWORD)v125 );
        }
LABEL_35:
        if ( v28 == (int *)(v123 + 4LL * (unsigned int)v125) )
          goto LABEL_42;
        goto LABEL_36;
      }
    }
  }
LABEL_2:
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 8);
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  sub_22B4A70(v7, v6, (__int64)&v126);
  if ( !(_DWORD)v128 )
  {
    v8 = (__int64)v127;
    v9 = (unsigned int)v129;
    goto LABEL_4;
  }
  v8 = (__int64)v127;
  v9 = (unsigned int)v129;
  v46 = &v127[v9];
  if ( v127 != &v127[v9] )
  {
    v47 = v127;
    while ( *v47 == -8192 || *v47 == -4096 )
    {
      if ( v46 == ++v47 )
        goto LABEL_4;
    }
    if ( v46 != v47 )
    {
      v48 = a2;
      v49 = *v47;
      v50 = *(unsigned int *)(a1 + 48);
      v51 = *(_QWORD *)(a1 + 32);
      if ( !(_DWORD)v50 )
        goto LABEL_76;
LABEL_66:
      v52 = (v50 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v53 = (__int64 *)(v51 + 16LL * v52);
      v54 = *v53;
      if ( v49 == *v53 )
      {
LABEL_67:
        v55 = *((_DWORD *)v53 + 2);
        v56 = *(unsigned int *)(a1 + 112);
        v57 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)v56 )
          goto LABEL_77;
        goto LABEL_68;
      }
      v91 = 1;
      while ( v54 != -4096 )
      {
        v92 = v91 + 1;
        v52 = (v50 - 1) & (v91 + v52);
        v53 = (__int64 *)(v51 + 16LL * v52);
        v54 = *v53;
        if ( v49 == *v53 )
          goto LABEL_67;
        v91 = v92;
      }
      while ( 1 )
      {
LABEL_76:
        v57 = *(_QWORD *)(a1 + 96);
        v55 = *(_DWORD *)(v51 + 16LL * (unsigned int)v50 + 8);
        v56 = *(unsigned int *)(a1 + 112);
        if ( (_DWORD)v56 )
        {
LABEL_68:
          v58 = (v56 - 1) & (37 * v55);
          v59 = *(_DWORD *)(v57 + 8LL * v58);
          if ( v55 == v59 )
            goto LABEL_69;
          v90 = 1;
          while ( v59 != -1 )
          {
            v58 = (v56 - 1) & (v90 + v58);
            v59 = *(_DWORD *)(v57 + 8LL * v58);
            if ( v55 == v59 )
              goto LABEL_69;
            ++v90;
          }
        }
LABEL_77:
        v60 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL);
        if ( v49 != *(_QWORD *)(v60 + 40) )
        {
          v117 = v48;
          sub_AA72C0(&v134, v49, 1);
          v61 = _mm_loadu_si128(&v134);
          v132 = 0;
          v62 = _mm_loadu_si128(&v135);
          v48 = v117;
          v130[0] = v61;
          v130[1] = v62;
          if ( v137 )
          {
            v137(v131, v136, 2);
            v60 = *(_QWORD *)&v130[0];
            v48 = v117;
            v133 = v138;
            if ( *(_QWORD *)&v130[0] )
              v60 = *(_QWORD *)&v130[0] - 24LL;
            v132 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v137;
            if ( v137 )
            {
              v137(v131, v131, 3);
              v48 = v117;
            }
          }
          else
          {
            v60 = *(_QWORD *)&v130[0];
            if ( *(_QWORD *)&v130[0] )
              v60 = *(_QWORD *)&v130[0] - 24LL;
          }
          if ( v140 )
          {
            v118 = v48;
            v140(v139, v139, 3);
            v48 = v118;
          }
          if ( v137 )
          {
            v119 = v48;
            v137(v136, v136, 3);
            v48 = v119;
          }
          v51 = *(_QWORD *)(a1 + 32);
          v50 = *(unsigned int *)(a1 + 48);
          v57 = *(_QWORD *)(a1 + 96);
          v56 = *(unsigned int *)(a1 + 112);
        }
        if ( (_DWORD)v50 )
        {
          v63 = (v50 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v64 = (__int64 *)(v51 + 16LL * v63);
          v65 = *v64;
          if ( v60 == *v64 )
          {
LABEL_90:
            if ( v64 != (__int64 *)(v51 + 16 * v50) )
              v109 = *((_DWORD *)v64 + 2);
          }
          else
          {
            v93 = 1;
            while ( v65 != -4096 )
            {
              v63 = (v50 - 1) & (v93 + v63);
              v120 = v93 + 1;
              v64 = (__int64 *)(v51 + 16LL * v63);
              v65 = *v64;
              if ( v60 == *v64 )
                goto LABEL_90;
              v93 = v120;
            }
          }
        }
        if ( (_DWORD)v56 )
        {
          v66 = (v56 - 1) & (37 * v109);
          v67 = (int *)(v57 + 8LL * v66);
          v68 = *v67;
          if ( *v67 == v109 )
          {
LABEL_94:
            if ( v67 != (int *)(v57 + 8 * v56) )
              v111 = v67[1];
          }
          else
          {
            v94 = 1;
            while ( v68 != -1 )
            {
              v95 = v94 + 1;
              v66 = (v56 - 1) & (v94 + v66);
              v67 = (int *)(v57 + 8LL * v66);
              v68 = *v67;
              if ( *v67 == v109 )
                goto LABEL_94;
              v94 = v95;
            }
          }
        }
        v69 = *(unsigned int *)(v48 + 144);
        v70 = *(_QWORD *)(v48 + 128);
        if ( (_DWORD)v69 )
        {
          v71 = (v69 - 1) & (37 * v111);
          v72 = (int *)(v70 + 8LL * v71);
          v73 = *v72;
          if ( *v72 == v111 )
          {
LABEL_98:
            if ( v72 != (int *)(v70 + 8 * v69) )
              v110 = v72[1];
          }
          else
          {
            v96 = 1;
            while ( v73 != -1 )
            {
              v97 = v96 + 1;
              v71 = (v69 - 1) & (v96 + v71);
              v72 = (int *)(v70 + 8LL * v71);
              v73 = *v72;
              if ( v111 == *v72 )
                goto LABEL_98;
              v96 = v97;
            }
          }
        }
        v74 = *(unsigned int *)(v48 + 80);
        v75 = *(_QWORD *)(v48 + 64);
        if ( (_DWORD)v74 )
        {
          v76 = (v74 - 1) & (37 * v110);
          v77 = (int *)(v75 + 16LL * v76);
          v78 = *v77;
          if ( v110 == *v77 )
          {
LABEL_102:
            if ( v77 != (int *)(v75 + 16 * v74) )
              v112 = *((_QWORD *)v77 + 1);
          }
          else
          {
            v98 = 1;
            while ( v78 != -1 )
            {
              v99 = v98 + 1;
              v76 = (v74 - 1) & (v98 + v76);
              v77 = (int *)(v75 + 16LL * v76);
              v78 = *v77;
              if ( v110 == *v77 )
                goto LABEL_102;
              v98 = v99;
            }
          }
        }
        v79 = *(unsigned int *)(v48 + 48);
        v80 = *(_QWORD *)(v48 + 32);
        v81 = *(_QWORD *)(v112 + 40);
        if ( (_DWORD)v79 )
        {
          v82 = (v79 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
          v83 = (__int64 *)(v80 + 16LL * v82);
          v84 = *v83;
          if ( v81 == *v83 )
          {
LABEL_106:
            if ( v83 != (__int64 *)(v80 + 16 * v79) )
              v108 = *((_DWORD *)v83 + 2);
          }
          else
          {
            v100 = 1;
            while ( v84 != -4096 )
            {
              v101 = v100 + 1;
              v82 = (v79 - 1) & (v100 + v82);
              v83 = (__int64 *)(v80 + 16LL * v82);
              v84 = *v83;
              if ( v81 == *v83 )
                goto LABEL_106;
              v100 = v101;
            }
          }
        }
        v85 = *(unsigned int *)(v48 + 112);
        v86 = *(_QWORD *)(v48 + 96);
        if ( (_DWORD)v85 )
        {
          v87 = (v85 - 1) & (37 * v108);
          v88 = (int *)(v86 + 8LL * v87);
          v89 = *v88;
          if ( *v88 == v108 )
          {
LABEL_110:
            if ( v88 != (int *)(v86 + 8 * v85) )
              v115 = v88[1];
          }
          else
          {
            v102 = 1;
            while ( v89 != -1 )
            {
              v103 = v102 + 1;
              v87 = (v85 - 1) & (v102 + v87);
              v88 = (int *)(v86 + 8LL * v87);
              v89 = *v88;
              if ( v108 == *v88 )
                goto LABEL_110;
              v102 = v103;
            }
          }
        }
        v116 = v48;
        *(_QWORD *)&v130[0] = __PAIR64__(v55, v115);
        sub_22B89D0((__int64)&v134, a1 + 120, (int *)v130, (int *)v130 + 1);
        *(_QWORD *)&v130[0] = __PAIR64__(v115, v55);
        sub_22B89D0((__int64)&v134, a1 + 88, (int *)v130, (int *)v130 + 1);
        v48 = v116;
LABEL_69:
        if ( ++v47 == v46 )
          goto LABEL_73;
        while ( *v47 == -4096 || *v47 == -8192 )
        {
          if ( v46 == ++v47 )
            goto LABEL_73;
        }
        if ( v46 == v47 )
        {
LABEL_73:
          v8 = (__int64)v127;
          v9 = (unsigned int)v129;
          break;
        }
        v50 = *(unsigned int *)(a1 + 48);
        v49 = *v47;
        v51 = *(_QWORD *)(a1 + 32);
        if ( (_DWORD)v50 )
          goto LABEL_66;
      }
    }
  }
LABEL_4:
  sub_C7D6A0(v8, v9 * 8, 8);
  return sub_C7D6A0(v123, 4LL * (unsigned int)v125, 4);
}
