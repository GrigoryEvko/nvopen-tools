// Function: sub_1DC3E40
// Address: 0x1dc3e40
//
__int64 __fastcall sub_1DC3E40(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 *v4; // rbx
  _QWORD *v5; // r9
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 *v8; // r14
  __int64 v9; // r12
  __int64 *v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 *v21; // rcx
  __int64 v22; // rsi
  bool v23; // cl
  __int64 v24; // r15
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 *v27; // rdx
  __int64 v28; // r12
  bool v29; // r14
  __int64 v30; // r13
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned int v39; // ecx
  __int64 *v40; // rdx
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 *v46; // rax
  __int64 v47; // r10
  __int64 v48; // r11
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 v52; // r10
  int v53; // r14d
  _QWORD *v54; // rdi
  __int64 v55; // r15
  __int64 v56; // r13
  unsigned int v57; // edx
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // r11
  _QWORD *v61; // rax
  int v62; // r12d
  _QWORD *v63; // rcx
  __int64 *v64; // rax
  int v65; // edx
  __int64 v66; // r12
  unsigned __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // r13
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rcx
  unsigned int v73; // r10d
  __int64 *v74; // rdx
  __int64 v75; // rsi
  __int64 v76; // rax
  _QWORD *v77; // rdi
  __int64 v78; // r14
  __int64 *v79; // rax
  __int64 v80; // r10
  __int64 *v81; // r11
  int v82; // r8d
  int v83; // edx
  int v84; // edi
  __int128 v85; // [rsp-20h] [rbp-2D0h]
  __int128 v86; // [rsp-20h] [rbp-2D0h]
  _QWORD *v87; // [rsp+0h] [rbp-2B0h]
  _QWORD *v88; // [rsp+10h] [rbp-2A0h]
  __int64 v89; // [rsp+10h] [rbp-2A0h]
  __int64 v90; // [rsp+20h] [rbp-290h]
  _QWORD *v91; // [rsp+28h] [rbp-288h]
  unsigned int v92; // [rsp+28h] [rbp-288h]
  __int64 v93; // [rsp+28h] [rbp-288h]
  char v94; // [rsp+36h] [rbp-27Ah]
  bool v95; // [rsp+37h] [rbp-279h]
  __int64 *v96; // [rsp+38h] [rbp-278h]
  __int64 *v97; // [rsp+40h] [rbp-270h]
  __int64 v98; // [rsp+48h] [rbp-268h]
  __int64 v99; // [rsp+50h] [rbp-260h]
  _QWORD *v100; // [rsp+50h] [rbp-260h]
  __int64 v101; // [rsp+50h] [rbp-260h]
  __int64 v102; // [rsp+58h] [rbp-258h]
  __int64 v103; // [rsp+58h] [rbp-258h]
  int v104; // [rsp+60h] [rbp-250h]
  _QWORD *v105; // [rsp+60h] [rbp-250h]
  __int64 v106; // [rsp+60h] [rbp-250h]
  __int64 v107; // [rsp+60h] [rbp-250h]
  _QWORD *v108; // [rsp+60h] [rbp-250h]
  __int64 v109; // [rsp+60h] [rbp-250h]
  __int64 v110; // [rsp+60h] [rbp-250h]
  _QWORD *v111; // [rsp+60h] [rbp-250h]
  _QWORD *v112; // [rsp+68h] [rbp-248h]
  _QWORD *v113; // [rsp+68h] [rbp-248h]
  __int64 v114; // [rsp+68h] [rbp-248h]
  _QWORD *v115; // [rsp+68h] [rbp-248h]
  _QWORD *v116; // [rsp+70h] [rbp-240h] BYREF
  __int64 v117; // [rsp+78h] [rbp-238h]
  _QWORD v118[70]; // [rsp+80h] [rbp-230h] BYREF

  do
  {
    v2 = a1[17];
    result = v2 + 32LL * *((unsigned int *)a1 + 36);
    v97 = (__int64 *)result;
    if ( v2 == result )
      return result;
    v94 = 0;
    v4 = (__int64 *)a1[17];
    v5 = a1;
    do
    {
      result = v4[1];
      v98 = result;
      if ( !result )
        goto LABEL_13;
      v19 = *(_QWORD *)result;
      v20 = *(_QWORD *)(result + 8);
      result = v5[12];
      v103 = v19;
      if ( !v20 )
      {
        v7 = 16LL * *(unsigned int *)(v19 + 48);
        v8 = (__int64 *)(result + v7);
        goto LABEL_5;
      }
      v6 = *(unsigned int *)(*(_QWORD *)v20 + 48LL);
      if ( (*(_QWORD *)(v5[5] + 8LL * ((unsigned int)v6 >> 6)) & (1LL << v6)) == 0 )
        goto LABEL_4;
      v21 = (__int64 *)(result + 16 * v6);
      v22 = *v21;
      v114 = *v21;
      v90 = v21[1];
      v23 = *v21 != 0 && *v21 != (_QWORD)&qword_4FC4510;
      v95 = v23;
      if ( !v90 && v23 )
      {
        v66 = v5[3];
        v67 = *(_QWORD *)(v22 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v67 )
        {
          v68 = *(_QWORD *)(v67 + 16);
          if ( v68 )
          {
            v69 = *(_QWORD *)(v68 + 24);
            goto LABEL_88;
          }
        }
        v76 = v5[2];
        v101 = v20;
        v111 = v5;
        v77 = *(_QWORD **)(v76 + 536);
        v116 = *(_QWORD **)(v22 + 8);
        v78 = *(unsigned int *)(v76 + 544);
        v79 = sub_1DC32D0(v77, (__int64)&v77[2 * v78], (__int64 *)&v116);
        v5 = v111;
        v20 = v101;
        if ( v81 == v79 )
        {
          if ( (_DWORD)v78 )
LABEL_96:
            v79 -= 2;
        }
        else if ( (*(_DWORD *)((*v79 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v79 >> 1) & 3) > (*(_DWORD *)(v67 + 24) | (unsigned int)(v80 >> 1) & 3) )
        {
          goto LABEL_96;
        }
        v69 = v79[1];
LABEL_88:
        v100 = v5;
        v109 = v20;
        sub_1E06620(v66);
        v70 = *(_QWORD *)(v66 + 1312);
        v90 = 0;
        v20 = v109;
        v5 = v100;
        v71 = *(unsigned int *)(v70 + 48);
        if ( (_DWORD)v71 )
        {
          v72 = *(_QWORD *)(v70 + 32);
          v73 = (v71 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
          v74 = (__int64 *)(v72 + 16LL * v73);
          v75 = *v74;
          if ( v69 == *v74 )
          {
LABEL_90:
            if ( v74 != (__int64 *)(v72 + 16 * v71) )
            {
              v90 = v74[1];
              goto LABEL_92;
            }
          }
          else
          {
            v83 = 1;
            while ( v75 != -8 )
            {
              v84 = v83 + 1;
              v73 = (v71 - 1) & (v83 + v73);
              v74 = (__int64 *)(v72 + 16LL * v73);
              v75 = *v74;
              if ( v69 == *v74 )
                goto LABEL_90;
              v83 = v84;
            }
          }
          v90 = 0;
        }
LABEL_92:
        *(_QWORD *)(v100[12] + 16LL * *(unsigned int *)(*(_QWORD *)v109 + 48LL) + 8) = v90;
        v24 = *(_QWORD *)(v103 + 64);
        result = v100[12];
        v99 = *(_QWORD *)(v103 + 72);
        if ( v24 == v99 )
          goto LABEL_38;
LABEL_20:
        v96 = v4;
        v25 = v20;
        while ( 1 )
        {
          v26 = result + 16LL * *(unsigned int *)(*(_QWORD *)v24 + 48LL);
          v27 = *(__int64 **)v26;
          if ( *(_QWORD *)v26 == v114 || !v27 )
            goto LABEL_35;
          if ( v27 == &qword_4FC4510 )
            goto LABEL_33;
          v28 = *(_QWORD *)(v26 + 8);
          if ( v28 )
          {
            v29 = v28 == v25;
            goto LABEL_26;
          }
          v33 = v5[3];
          if ( (v27[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            break;
          v34 = *(_QWORD *)((v27[1] & 0xFFFFFFFFFFFFFFF8LL) + 16);
          if ( !v34 )
            break;
          v35 = *(_QWORD *)(v34 + 24);
LABEL_46:
          v91 = v5;
          v106 = v35;
          sub_1E06620(v33);
          v36 = *(_QWORD *)(v33 + 1312);
          v5 = v91;
          v37 = *(unsigned int *)(v36 + 48);
          if ( !(_DWORD)v37 )
            goto LABEL_67;
          v38 = *(_QWORD *)(v36 + 32);
          v39 = (v37 - 1) & (((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4));
          v40 = (__int64 *)(v38 + 16LL * v39);
          v41 = *v40;
          if ( *v40 == v106 )
          {
LABEL_48:
            if ( v40 == (__int64 *)(v38 + 16 * v37) )
            {
LABEL_67:
              *(_QWORD *)(v26 + 8) = 0;
              v4 = v96;
              sub_1E06620(v91[3]);
              v5 = v91;
LABEL_42:
              result = v5[12];
LABEL_4:
              v7 = 16LL * *(unsigned int *)(v103 + 48);
              v8 = (__int64 *)(result + v7);
LABEL_5:
              v9 = *v4;
              v112 = v5;
              v10 = (__int64 *)v5[4];
              v11 = (__int64 *)(*(_QWORD *)(v5[2] + 392LL) + v7);
              v12 = *v11;
              v102 = v11[1];
              v104 = *(_DWORD *)(*v4 + 72);
              v13 = sub_145CDC0(0x10u, v10);
              v5 = v112;
              v16 = v13;
              if ( v13 )
              {
                *(_QWORD *)(v13 + 8) = v12;
                *(_DWORD *)v13 = v104;
              }
              v17 = *(unsigned int *)(v9 + 72);
              if ( (unsigned int)v17 >= *(_DWORD *)(v9 + 76) )
              {
                v10 = (__int64 *)(v9 + 80);
                sub_16CD150(v9 + 64, (const void *)(v9 + 80), 0, 8, v15, (int)v112);
                v17 = *(unsigned int *)(v9 + 72);
                v5 = v112;
              }
              v18 = *(_QWORD *)(v9 + 64);
              *(_QWORD *)(v18 + 8 * v17) = v16;
              ++*(_DWORD *)(v9 + 72);
              result = v4[2];
              v4[3] = v16;
              v4[1] = 0;
              if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( v16 )
                {
                  v117 = result;
                  v118[0] = v16;
                  *((_QWORD *)&v85 + 1) = result;
                  *(_QWORD *)&v85 = v12;
                  v113 = v5;
                  v116 = (_QWORD *)v12;
                  result = sub_1DB8610(v9, (__int64)v10, v18, v14, v15, (__int64)v5, v85, v16);
                  v5 = v113;
                }
              }
              else
              {
                if ( v16 )
                {
                  v118[0] = v16;
                  v117 = v102;
                  *((_QWORD *)&v86 + 1) = v102;
                  *(_QWORD *)&v86 = v12;
                  v115 = v5;
                  v116 = (_QWORD *)v12;
                  sub_1DB8610(v9, (__int64)v10, v18, v14, v15, (__int64)v5, v86, v16);
                  v5 = v115;
                }
                result = v98;
                *v8 = v16;
                v8[1] = v98;
              }
LABEL_12:
              v94 = 1;
              goto LABEL_13;
            }
            v28 = v40[1];
            v29 = v25 == v28 || v28 == 0;
          }
          else
          {
            v65 = 1;
            while ( v41 != -8 )
            {
              v82 = v65 + 1;
              v39 = (v37 - 1) & (v65 + v39);
              v40 = (__int64 *)(v38 + 16LL * v39);
              v41 = *v40;
              if ( v106 == *v40 )
                goto LABEL_48;
              v65 = v82;
            }
            v29 = 1;
          }
          *(_QWORD *)(v26 + 8) = v28;
LABEL_26:
          v30 = v5[3];
          v105 = v5;
          sub_1E06620(v30);
          v5 = v105;
          if ( v29 || v25 == *(_QWORD *)(v28 + 8) )
          {
            v4 = v96;
            goto LABEL_42;
          }
          if ( *(_QWORD *)(v25 + 8) == v28 || *(_DWORD *)(v25 + 16) >= *(_DWORD *)(v28 + 16) )
          {
            result = v105[12];
            goto LABEL_35;
          }
          v31 = *(_QWORD *)(v30 + 1312);
          if ( *(_BYTE *)(v31 + 72) )
          {
            result = v105[12];
            if ( *(_DWORD *)(v28 + 48) >= *(_DWORD *)(v25 + 48) )
            {
LABEL_32:
              if ( *(_DWORD *)(v28 + 52) <= *(_DWORD *)(v25 + 52) )
                goto LABEL_33;
            }
LABEL_35:
            v24 += 8;
            if ( v99 == v24 )
              goto LABEL_36;
            continue;
          }
          v42 = *(_DWORD *)(v31 + 76) + 1;
          *(_DWORD *)(v31 + 76) = v42;
          if ( v42 > 0x20 )
          {
            HIDWORD(v117) = 32;
            v116 = v118;
            v49 = *(_QWORD *)(v31 + 56);
            if ( v49 )
            {
              v50 = *(_QWORD *)(v49 + 24);
              v51 = v24;
              v52 = v31;
              v53 = 1;
              v118[0] = *(_QWORD *)(v31 + 56);
              v54 = v118;
              v55 = v28;
              v56 = v25;
              v118[1] = v50;
              v57 = 1;
              LODWORD(v117) = 1;
              *(_DWORD *)(v49 + 48) = 0;
              do
              {
                while ( 1 )
                {
                  v62 = v53++;
                  v63 = &v54[2 * v57 - 2];
                  v64 = (__int64 *)v63[1];
                  if ( v64 != *(__int64 **)(*v63 + 32LL) )
                    break;
                  --v57;
                  *(_DWORD *)(*v63 + 52LL) = v62;
                  LODWORD(v117) = v57;
                  if ( !v57 )
                    goto LABEL_75;
                }
                v58 = *v64;
                v63[1] = v64 + 1;
                v59 = (unsigned int)v117;
                v60 = *(_QWORD *)(v58 + 24);
                if ( (unsigned int)v117 >= HIDWORD(v117) )
                {
                  v87 = v5;
                  v89 = v52;
                  v93 = v51;
                  v110 = *(_QWORD *)(v58 + 24);
                  sub_16CD150((__int64)&v116, v118, 0, 16, v51, (int)v5);
                  v54 = v116;
                  v59 = (unsigned int)v117;
                  v5 = v87;
                  v52 = v89;
                  v51 = v93;
                  v60 = v110;
                }
                v61 = &v54[2 * v59];
                *v61 = v58;
                v61[1] = v60;
                LODWORD(v117) = v117 + 1;
                v57 = v117;
                *(_DWORD *)(v58 + 48) = v62;
                v54 = v116;
              }
              while ( v57 );
LABEL_75:
              v28 = v55;
              *(_BYTE *)(v52 + 72) = 1;
              v25 = v56;
              v24 = v51;
              *(_DWORD *)(v52 + 76) = 0;
              if ( v54 != v118 )
              {
                v108 = v5;
                _libc_free((unsigned __int64)v54);
                v5 = v108;
              }
            }
            result = v5[12];
            if ( *(_DWORD *)(v28 + 48) >= *(_DWORD *)(v25 + 48) )
              goto LABEL_32;
            v24 += 8;
            if ( v99 == v24 )
              goto LABEL_36;
          }
          else
          {
            do
            {
              v43 = v28;
              v28 = *(_QWORD *)(v28 + 8);
            }
            while ( v28 && *(_DWORD *)(v25 + 16) <= *(_DWORD *)(v28 + 16) );
            result = v105[12];
            if ( v25 == v43 )
            {
LABEL_33:
              v4 = v96;
              goto LABEL_4;
            }
            v24 += 8;
            if ( v99 == v24 )
            {
LABEL_36:
              v4 = v96;
              goto LABEL_37;
            }
          }
        }
        v44 = v5[2];
        v88 = v5;
        v45 = *(_QWORD **)(v44 + 536);
        v116 = (_QWORD *)v27[1];
        v92 = *(_DWORD *)(v44 + 544);
        v107 = (__int64)&v45[2 * v92];
        v46 = sub_1DC32D0(v45, v107, (__int64 *)&v116);
        v5 = v88;
        if ( (__int64 *)v107 == v46 )
        {
          if ( v92 )
LABEL_65:
            v46 -= 2;
        }
        else if ( (*(_DWORD *)((*v46 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v46 >> 1) & 3) > (*(_DWORD *)(v48 + 24) | (unsigned int)(v47 >> 1) & 3) )
        {
          goto LABEL_65;
        }
        v35 = v46[1];
        goto LABEL_46;
      }
      v24 = *(_QWORD *)(v103 + 64);
      v99 = *(_QWORD *)(v103 + 72);
      if ( v99 != v24 )
        goto LABEL_20;
LABEL_37:
      if ( !v95 )
        goto LABEL_13;
LABEL_38:
      v32 = *(unsigned int *)(v103 + 48);
      v4[3] = v114;
      if ( (v4[2] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        result += 16 * v32;
        if ( *(_QWORD *)result != v114 )
        {
          *(_QWORD *)result = v114;
          *(_QWORD *)(result + 8) = v90;
          goto LABEL_12;
        }
      }
LABEL_13:
      v4 += 4;
    }
    while ( v97 != v4 );
    a1 = v5;
  }
  while ( v94 );
  return result;
}
