// Function: sub_3236930
// Address: 0x3236930
//
void __fastcall sub_3236930(__int64 a1, __int64 a2, int *a3, size_t a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  int v13; // r11d
  int v14; // edx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r9
  _QWORD *v21; // r14
  __int64 v22; // rax
  int v23; // edx
  _QWORD *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int16 v27; // ax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r9
  bool v31; // zf
  _QWORD *v32; // r15
  _BYTE *v33; // r14
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r14
  __int64 *v38; // rbx
  __int64 *j; // r13
  __int64 v40; // r12
  char v41; // al
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 k; // r14
  unsigned __int64 v48; // rdi
  __int64 v49; // r12
  unsigned __int64 v50; // rbx
  _BYTE *v51; // r12
  _QWORD *v52; // r13
  unsigned __int16 v53; // ax
  __int64 v54; // rdi
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  __int64 v57; // rax
  _QWORD *v58; // rax
  _BYTE *v59; // r14
  __int64 v60; // rdx
  __int64 v61; // rdx
  _BYTE *v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // r13
  __int64 v66; // r14
  _QWORD *v67; // r12
  int v68; // eax
  __int64 v69; // r13
  __int64 v70; // r12
  unsigned __int64 v71; // rdi
  __int64 v72; // r9
  unsigned __int64 v73; // rsi
  unsigned __int64 i; // r9
  int v75; // ecx
  __int64 v76; // rdi
  __int64 v77; // r8
  int v78; // ecx
  unsigned int v79; // edx
  __int64 *v80; // rax
  __int64 v81; // r10
  unsigned __int8 v82; // al
  __int64 v83; // rdx
  unsigned __int64 v84; // rbx
  _BYTE *v85; // r12
  _QWORD *v86; // r13
  _QWORD *v87; // rdx
  _QWORD *v88; // rax
  unsigned __int64 v89; // rdi
  int v90; // eax
  _QWORD *v91; // rdx
  int v92; // eax
  int v93; // r11d
  int v94; // edi
  int v95; // edi
  __int64 v96; // r9
  unsigned int v97; // eax
  __int64 v98; // r8
  __int64 *v99; // r10
  int v100; // esi
  __int64 *v101; // rcx
  int v102; // esi
  int v103; // esi
  __int64 v104; // rdi
  unsigned int v105; // r12d
  __int64 v106; // rcx
  __int64 *v107; // r9
  int v108; // eax
  __int64 *v109; // r8
  _QWORD *v110; // [rsp+0h] [rbp-B0h]
  int v111; // [rsp+0h] [rbp-B0h]
  __int64 v112; // [rsp+8h] [rbp-A8h]
  int v113; // [rsp+10h] [rbp-A0h]
  _QWORD *v114; // [rsp+10h] [rbp-A0h]
  int v115; // [rsp+18h] [rbp-98h]
  __int64 v116; // [rsp+18h] [rbp-98h]
  int v117; // [rsp+18h] [rbp-98h]
  int v118; // [rsp+18h] [rbp-98h]
  int v119; // [rsp+20h] [rbp-90h]
  _QWORD *v120; // [rsp+20h] [rbp-90h]
  unsigned __int64 v123; // [rsp+30h] [rbp-80h]
  _QWORD *v124; // [rsp+38h] [rbp-78h]
  __int64 *v125; // [rsp+38h] [rbp-78h]
  __int64 v126; // [rsp+38h] [rbp-78h]
  int v127; // [rsp+38h] [rbp-78h]
  unsigned __int64 v130; // [rsp+58h] [rbp-58h] BYREF
  _BYTE *v131; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v132; // [rsp+68h] [rbp-48h]
  _BYTE v133[64]; // [rsp+70h] [rbp-40h] BYREF

  v7 = a1;
  if ( !*(_DWORD *)(a1 + 3648) || !*(_BYTE *)(a1 + 4872) )
  {
    v8 = *(_DWORD *)(a1 + 3600);
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 3584);
      v10 = (v8 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == a6 )
      {
LABEL_5:
        sub_324A0C0(a2, a5, v11[1]);
        return;
      }
      v124 = 0;
      v13 = 1;
      while ( v12 != -4096 )
      {
        if ( v12 == -8192 )
        {
          if ( v124 )
            v11 = v124;
          v124 = v11;
        }
        v10 = (v8 - 1) & (v13 + v10);
        v11 = (_QWORD *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( *v11 == a6 )
          goto LABEL_5;
        ++v13;
      }
      if ( v124 )
        v11 = v124;
      ++*(_QWORD *)(v7 + 3576);
      v125 = v11;
      v14 = *(_DWORD *)(v7 + 3592) + 1;
      if ( 4 * v14 < 3 * v8 )
      {
        if ( v8 - *(_DWORD *)(v7 + 3596) - v14 > v8 >> 3 )
        {
LABEL_13:
          *(_DWORD *)(v7 + 3592) = v14;
          if ( *v125 != -4096 )
            --*(_DWORD *)(v7 + 3596);
          v15 = v7 + 3080;
          *v125 = a6;
          v125[1] = 0;
          *(_QWORD *)(v7 + 5384) = v7 + 5136;
          *(_BYTE *)(v7 + 4872) = 0;
          v112 = v7 + 5136;
          v115 = *(_DWORD *)(v7 + 3648);
          v16 = sub_3222C60(v7, a2);
          v17 = *(_QWORD *)(v7 + 8);
          v18 = v16;
          v113 = *(_DWORD *)(v7 + 3680);
          *(_DWORD *)(v7 + 3680) = v113 + 1;
          v119 = v17;
          v19 = sub_22077B0(0x1B0u);
          v21 = (_QWORD *)v19;
          if ( v19 )
            sub_3247880(v19, a2, v119, v7, v7 + 3080, v113, v18);
          v22 = *(unsigned int *)(v7 + 3648);
          v23 = v22;
          if ( *(_DWORD *)(v7 + 3652) <= (unsigned int)v22 )
          {
            v120 = v21;
            v87 = (_QWORD *)sub_C8D7D0(v7 + 3640, v7 + 3656, 0, 0x10u, (unsigned __int64 *)&v131, v20);
            v88 = &v87[2 * *(unsigned int *)(v7 + 3648)];
            if ( v88 )
            {
              *v88 = v21;
              v88[1] = a6;
              v120 = 0;
            }
            v110 = v87;
            sub_3226780(v7 + 3640, v87);
            v89 = *(_QWORD *)(v7 + 3640);
            v90 = (int)v131;
            v91 = v110;
            if ( v7 + 3656 != v89 )
            {
              v111 = (int)v131;
              v114 = v91;
              _libc_free(v89);
              v90 = v111;
              v91 = v114;
            }
            ++*(_DWORD *)(v7 + 3648);
            *(_QWORD *)(v7 + 3640) = v91;
            *(_DWORD *)(v7 + 3652) = v90;
          }
          else
          {
            v120 = v21;
            v24 = (_QWORD *)(*(_QWORD *)(v7 + 3640) + 16 * v22);
            if ( v24 )
            {
              v120 = 0;
              *v24 = v21;
              v24[1] = a6;
              v23 = *(_DWORD *)(v7 + 3648);
            }
            *(_DWORD *)(v7 + 3648) = v23 + 1;
          }
          v25 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 80) + 16LL);
          LODWORD(v131) = 65541;
          sub_3249A20(v21, v21 + 2, 19, v131, v25);
          v26 = sub_3220A40(a3, a4);
          v21[49] = v26;
          v125[1] = v26;
          v123 = v26;
          if ( *(_BYTE *)(v7 + 3769) )
          {
            if ( (unsigned __int16)sub_3220AA0(v7) > 4u )
            {
              if ( *(_QWORD *)(v7 + 3072) )
                sub_324AD70(v21, v21 + 1, 27, *(_QWORD *)(v7 + 3064), *(_QWORD *)(v7 + 3072));
              sub_324AD70(
                v21,
                v21 + 1,
                118,
                *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 8) + 200LL) + 1072LL),
                *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 8) + 200LL) + 1080LL));
            }
            v27 = sub_3220AA0(v7);
            v28 = *(_QWORD *)(v7 + 8);
            if ( v27 > 4u )
              v29 = *(_QWORD *)(sub_31DA6B0(v28) + 232);
            else
              v29 = *(_QWORD *)(sub_31DA6B0(v28) + 240);
            v21[7] = v29;
          }
          else
          {
            v53 = sub_3220AA0(v7);
            v54 = *(_QWORD *)(v7 + 8);
            if ( v53 > 4u )
            {
              v57 = sub_31DA6B0(v54);
              v56 = sub_E89A10(v57, ".debug_info", v123);
            }
            else
            {
              v55 = sub_31DA6B0(v54);
              v56 = sub_E89A10(v55, ".debug_types", v123);
            }
            v21[7] = v56;
            sub_3735D50(a2, v21 + 1);
          }
          if ( *(_BYTE *)(v7 + 3770) && !*(_BYTE *)(v7 + 3769) )
            sub_324ACD0(v21);
          v21[50] = sub_3251340(v21, a6);
          if ( v115 )
            goto LABEL_64;
          v30 = *(unsigned int *)(v7 + 3648);
          v131 = v133;
          v132 = 0x100000000LL;
          if ( (_DWORD)v30 )
          {
            v58 = (_QWORD *)(v7 + 3656);
            if ( *(_QWORD *)(v7 + 3640) == v7 + 3656 )
            {
              v59 = v133;
              v60 = 1;
              if ( (_DWORD)v30 != 1 )
              {
                v117 = v30;
                v59 = (_BYTE *)sub_C8D7D0((__int64)&v131, (__int64)v133, (unsigned int)v30, 0x10u, &v130, v30);
                sub_3226780((__int64)&v131, v59);
                v68 = v130;
                LODWORD(v30) = v117;
                if ( v131 != v133 )
                {
                  v118 = v130;
                  v127 = v30;
                  _libc_free((unsigned __int64)v131);
                  v68 = v118;
                  LODWORD(v30) = v127;
                }
                HIDWORD(v132) = v68;
                v60 = *(unsigned int *)(v7 + 3648);
                v131 = v59;
                v58 = *(_QWORD **)(v7 + 3640);
              }
              v61 = 16 * v60;
              if ( v61 )
              {
                v62 = &v59[v61];
                do
                {
                  if ( v59 )
                  {
                    *(_QWORD *)v59 = *v58;
                    v63 = v58[1];
                    *v58 = 0;
                    *((_QWORD *)v59 + 1) = v63;
                  }
                  v59 += 16;
                  v58 += 2;
                }
                while ( v59 != v62 );
                v64 = *(unsigned int *)(v7 + 3648);
                v65 = *(_QWORD *)(v7 + 3640);
                LODWORD(v132) = v30;
                v64 *= 16;
                v66 = v65 + v64;
                if ( v65 != v65 + v64 )
                {
                  do
                  {
                    v67 = *(_QWORD **)(v66 - 16);
                    v66 -= 16;
                    if ( v67 )
                    {
                      *v67 = &unk_4A35D40;
                      sub_32478E0(v67);
                      j_j___libc_free_0((unsigned __int64)v67);
                    }
                  }
                  while ( v65 != v66 );
                  v15 = v7 + 3080;
                }
              }
              else
              {
                LODWORD(v132) = v30;
              }
            }
            else
            {
              v131 = *(_BYTE **)(v7 + 3640);
              v132 = __PAIR64__(*(_DWORD *)(v7 + 3652), v30);
              *(_QWORD *)(v7 + 3640) = v58;
              *(_DWORD *)(v7 + 3652) = 0;
            }
          }
          v31 = *(_BYTE *)(v7 + 4872) == 0;
          *(_DWORD *)(v7 + 3648) = 0;
          if ( !v31 )
          {
            sub_3226830(v7 + 5232);
            v69 = *(_QWORD *)(v7 + 5264);
            v70 = v69 + ((unsigned __int64)*(unsigned int *)(v7 + 5272) << 6);
            while ( v69 != v70 )
            {
              while ( 1 )
              {
                v71 = *(_QWORD *)(v70 - 32);
                v70 -= 64;
                if ( !v71 )
                  break;
                j_j___libc_free_0(v71);
                if ( v69 == v70 )
                  goto LABEL_99;
              }
            }
LABEL_99:
            v72 = (unsigned int)v132;
            v73 = (unsigned __int64)v131;
            *(_DWORD *)(v7 + 5272) = 0;
            for ( i = v73 + 16 * v72; v73 != i; v73 += 16LL )
            {
              v75 = *(_DWORD *)(v7 + 3600);
              v76 = *(_QWORD *)(v73 + 8);
              v77 = *(_QWORD *)(v7 + 3584);
              if ( v75 )
              {
                v78 = v75 - 1;
                v79 = v78 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                v80 = (__int64 *)(v77 + 16LL * v79);
                v81 = *v80;
                if ( v76 == *v80 )
                {
LABEL_102:
                  *v80 = -8192;
                  --*(_DWORD *)(v7 + 3592);
                  ++*(_DWORD *)(v7 + 3596);
                }
                else
                {
                  v92 = 1;
                  while ( v81 != -4096 )
                  {
                    v93 = v92 + 1;
                    v79 = v78 & (v92 + v79);
                    v80 = (__int64 *)(v77 + 16LL * v79);
                    v81 = *v80;
                    if ( v76 == *v80 )
                      goto LABEL_102;
                    v92 = v93;
                  }
                }
              }
            }
            *(_QWORD *)(v7 + 5384) = v7 + 4888;
            sub_32507E0(a2, a5, a6);
            v82 = *(_BYTE *)(a6 - 16);
            if ( (v82 & 2) != 0 )
              v83 = *(_QWORD *)(a6 - 32);
            else
              v83 = a6 - 16 - 8LL * ((v82 >> 2) & 0xF);
            sub_3248280(a2, *(_QWORD *)(v83 + 8), a6, a5);
            v84 = (unsigned __int64)v131;
            v85 = &v131[16 * (unsigned int)v132];
            if ( v131 != v85 )
            {
              do
              {
                v86 = (_QWORD *)*((_QWORD *)v85 - 2);
                v85 -= 16;
                if ( v86 )
                {
                  *v86 = &unk_4A35D40;
                  sub_32478E0(v86);
                  j_j___libc_free_0((unsigned __int64)v86);
                }
              }
              while ( (_BYTE *)v84 != v85 );
              v85 = v131;
            }
            if ( v85 != v133 )
              _libc_free((unsigned __int64)v85);
            goto LABEL_65;
          }
          v32 = v131;
          v33 = &v131[16 * (unsigned int)v132];
          if ( v33 == v131 )
          {
LABEL_42:
            v36 = *(_QWORD *)(v7 + 5264);
            v126 = v36 + ((unsigned __int64)*(unsigned int *)(v7 + 5272) << 6);
            if ( v36 != v126 )
            {
              v116 = v7;
              v37 = *(_QWORD *)(v7 + 5264);
              do
              {
                v38 = *(__int64 **)(v37 + 40);
                for ( j = *(__int64 **)(v37 + 32); v38 != j; ++j )
                {
                  v40 = *j;
                  v41 = *(_BYTE *)(*j + 16);
                  if ( v41 != 1 )
                  {
                    if ( v41 )
                      abort();
                    v42 = *(_QWORD *)(v40 + 8);
                    v43 = sub_37236D0(v42);
                    v31 = *(_BYTE *)(v40 + 16) == 1;
                    *(_QWORD *)(v40 + 24) = v43;
                    *(_QWORD *)(v40 + 32) = v44;
                    v45 = *(unsigned int *)(v42 + 16);
                    if ( !v31 )
                      *(_BYTE *)(v40 + 16) = 1;
                    *(_QWORD *)(v40 + 8) = v45;
                  }
                }
                v37 += 64;
              }
              while ( v126 != v37 );
              v7 = v116;
            }
            sub_32365A0((__int64 *)(v7 + 4888), v112);
            sub_3226830(v7 + 5232);
            v46 = *(_QWORD *)(v7 + 5264);
            for ( k = v46 + ((unsigned __int64)*(unsigned int *)(v7 + 5272) << 6); v46 != k; k -= 64 )
            {
              v48 = *(_QWORD *)(k - 32);
              if ( v48 )
                j_j___libc_free_0(v48);
            }
            *(_QWORD *)(v7 + 5384) = v7 + 4888;
            v49 = (unsigned int)v132;
            *(_DWORD *)(v7 + 5272) = 0;
            v50 = (unsigned __int64)v131;
            v51 = &v131[16 * v49];
            if ( v131 != v51 )
            {
              do
              {
                v52 = (_QWORD *)*((_QWORD *)v51 - 2);
                v51 -= 16;
                if ( v52 )
                {
                  *v52 = &unk_4A35D40;
                  sub_32478E0(v52);
                  j_j___libc_free_0((unsigned __int64)v52);
                }
              }
              while ( (_BYTE *)v50 != v51 );
              v51 = v131;
            }
            if ( v51 != v133 )
              _libc_free((unsigned __int64)v51);
LABEL_64:
            sub_324A0C0(a2, a5, v123);
LABEL_65:
            if ( v120 )
            {
              *v120 = &unk_4A35D40;
              sub_32478E0(v120);
              j_j___libc_free_0((unsigned __int64)v120);
            }
            return;
          }
          while ( 1 )
          {
            sub_3244F20(v15, *v32);
            sub_3244DB0(v15, *v32, *(unsigned __int8 *)(v7 + 3769));
            if ( (unsigned __int16)sub_3220AA0(v7) <= 4u || *(_DWORD *)(v7 + 3764) != 3 )
              goto LABEL_37;
            v34 = *v32;
            v35 = v7 + 4888;
            if ( *(_BYTE *)(v7 + 3769) )
            {
              sub_3723980(v35, v34);
              v32 += 2;
              if ( v33 == (_BYTE *)v32 )
                goto LABEL_42;
            }
            else
            {
              sub_37238A0(v35, v34);
LABEL_37:
              v32 += 2;
              if ( v33 == (_BYTE *)v32 )
                goto LABEL_42;
            }
          }
        }
        sub_322E110(v7 + 3576, v8);
        v102 = *(_DWORD *)(v7 + 3600);
        if ( v102 )
        {
          v103 = v102 - 1;
          v104 = *(_QWORD *)(v7 + 3584);
          v105 = v103 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
          v125 = (__int64 *)(v104 + 16LL * v105);
          v106 = *v125;
          v14 = *(_DWORD *)(v7 + 3592) + 1;
          if ( *v125 != a6 )
          {
            v107 = (__int64 *)(v104 + 16LL * v105);
            v108 = 1;
            v109 = 0;
            while ( v106 != -4096 )
            {
              if ( !v109 && v106 == -8192 )
                v109 = v107;
              v105 = v103 & (v108 + v105);
              v107 = (__int64 *)(v104 + 16LL * v105);
              v106 = *v107;
              if ( *v107 == a6 )
              {
                v125 = (__int64 *)(v104 + 16LL * v105);
                goto LABEL_13;
              }
              ++v108;
            }
            if ( !v109 )
              v109 = v107;
            v125 = v109;
          }
          goto LABEL_13;
        }
LABEL_152:
        JUMPOUT(0x4352A8);
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 3576);
    }
    sub_322E110(v7 + 3576, 2 * v8);
    v94 = *(_DWORD *)(v7 + 3600);
    if ( v94 )
    {
      v95 = v94 - 1;
      v96 = *(_QWORD *)(v7 + 3584);
      v97 = v95 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
      v125 = (__int64 *)(v96 + 16LL * v97);
      v98 = *v125;
      v14 = *(_DWORD *)(v7 + 3592) + 1;
      if ( *v125 != a6 )
      {
        v99 = (__int64 *)(v96 + 16LL * (v95 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
        v100 = 1;
        v101 = 0;
        while ( v98 != -4096 )
        {
          if ( v98 == -8192 && !v101 )
            v101 = v99;
          v97 = v95 & (v100 + v97);
          v99 = (__int64 *)(v96 + 16LL * v97);
          v98 = *v99;
          if ( *v99 == a6 )
          {
            v125 = (__int64 *)(v96 + 16LL * v97);
            goto LABEL_13;
          }
          ++v100;
        }
        if ( !v101 )
          v101 = v99;
        v125 = v101;
      }
      goto LABEL_13;
    }
    goto LABEL_152;
  }
}
