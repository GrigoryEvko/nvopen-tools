// Function: sub_1F61D20
// Address: 0x1f61d20
//
__int64 __fastcall sub_1F61D20(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r15
  __int64 v11; // rdi
  unsigned int v12; // eax
  unsigned int v13; // ecx
  __int64 *v14; // r14
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // r9
  __int64 v18; // rdi
  unsigned int v19; // ecx
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // r14
  unsigned __int64 v23; // rbx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  _QWORD **v28; // rdi
  __int64 v29; // r14
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // ecx
  __int64 *v34; // rdx
  __int64 v35; // r9
  int v36; // r14d
  unsigned int v37; // esi
  __int64 v38; // rdi
  unsigned int v39; // ecx
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // rdx
  _QWORD *v42; // rbx
  _QWORD *v43; // r12
  __int64 v44; // rax
  unsigned __int64 *v45; // rax
  unsigned __int64 *v46; // r13
  int v48; // r10d
  __int64 *v49; // rdx
  int v50; // ecx
  int v51; // r11d
  unsigned __int64 *v52; // r10
  int v53; // ecx
  int v54; // ecx
  int v55; // r10d
  __int64 *v56; // r9
  int v57; // ecx
  int v58; // ecx
  int v59; // eax
  int v60; // edi
  __int64 v61; // rsi
  __int64 v62; // rdx
  unsigned __int64 v63; // r8
  int v64; // r10d
  unsigned __int64 *v65; // r9
  unsigned int v66; // eax
  __int64 v67; // r8
  int v68; // r10d
  __int64 *v69; // r9
  int v70; // r9d
  __int64 v71; // r15
  __int64 *v72; // r8
  __int64 v73; // rsi
  int v74; // eax
  int v75; // esi
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rdi
  int v79; // r9d
  __int64 *v80; // r8
  int v81; // eax
  int v82; // edx
  __int64 v83; // rdi
  unsigned __int64 *v84; // r8
  __int64 v85; // r15
  int v86; // r9d
  unsigned __int64 v87; // rsi
  int v88; // edx
  int v89; // edx
  __int64 v90; // rdi
  int v91; // r9d
  __int64 v92; // rax
  __int64 v93; // rsi
  int v94; // edx
  int v95; // r10d
  int v96; // edi
  int v97; // edi
  __int64 v98; // r8
  __int64 v99; // rdx
  int v100; // ecx
  unsigned __int64 v101; // rsi
  int v102; // r10d
  unsigned __int64 *v103; // r11
  int v104; // ecx
  int v105; // edx
  int v106; // edx
  __int64 v107; // r9
  unsigned __int64 *v108; // rsi
  __int64 v109; // r15
  int v110; // edi
  unsigned __int64 v111; // r8
  int v112; // r10d
  unsigned __int64 *v113; // r9
  unsigned int v114; // [rsp+Ch] [rbp-64h]
  __int64 v115; // [rsp+18h] [rbp-58h]
  __int64 v116; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v117; // [rsp+28h] [rbp-48h]
  int v118; // [rsp+30h] [rbp-40h]
  int v119; // [rsp+34h] [rbp-3Ch]
  unsigned int v120; // [rsp+38h] [rbp-38h]

  sub_14DDFC0((__int64)&v116, a1);
  v3 = *(_QWORD *)(a1 + 80);
  v115 = a1 + 72;
  if ( v3 != a1 + 72 )
  {
    while ( 1 )
    {
      v22 = v3 - 24;
      if ( !v3 )
        v22 = 0;
      v23 = sub_157EBA0(v22);
      if ( *(_BYTE *)(v23 + 16) != 29 )
        goto LABEL_14;
      if ( !v120 )
        break;
      v24 = (v120 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v25 = &v117[2 * v24];
      v26 = *v25;
      if ( *v25 != v22 )
      {
        v48 = 1;
        v49 = 0;
        while ( v26 != -8 )
        {
          if ( v26 == -16 && !v49 )
            v49 = v25;
          v24 = (v120 - 1) & (v48 + v24);
          v25 = &v117[2 * v24];
          v26 = *v25;
          if ( *v25 == v22 )
            goto LABEL_20;
          ++v48;
        }
        if ( !v49 )
          v49 = v25;
        ++v116;
        v50 = v118 + 1;
        if ( 4 * (v118 + 1) < 3 * v120 )
        {
          if ( v120 - v119 - v50 <= v120 >> 3 )
          {
            sub_14DDDA0((__int64)&v116, v120);
            if ( !v120 )
            {
LABEL_200:
              ++v118;
              BUG();
            }
            v70 = 1;
            LODWORD(v71) = (v120 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v72 = 0;
            v50 = v118 + 1;
            v49 = &v117[2 * (unsigned int)v71];
            v73 = *v49;
            if ( *v49 != v22 )
            {
              while ( v73 != -8 )
              {
                if ( !v72 && v73 == -16 )
                  v72 = v49;
                v71 = (v120 - 1) & ((_DWORD)v71 + v70);
                v49 = &v117[2 * v71];
                v73 = *v49;
                if ( *v49 == v22 )
                  goto LABEL_52;
                ++v70;
              }
              if ( v72 )
                v49 = v72;
            }
          }
          goto LABEL_52;
        }
LABEL_82:
        sub_14DDDA0((__int64)&v116, 2 * v120);
        if ( !v120 )
          goto LABEL_200;
        v66 = (v120 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v50 = v118 + 1;
        v49 = &v117[2 * v66];
        v67 = *v49;
        if ( *v49 != v22 )
        {
          v68 = 1;
          v69 = 0;
          while ( v67 != -8 )
          {
            if ( v67 == -16 && !v69 )
              v69 = v49;
            v66 = (v120 - 1) & (v68 + v66);
            v49 = &v117[2 * v66];
            v67 = *v49;
            if ( *v49 == v22 )
              goto LABEL_52;
            ++v68;
          }
          if ( v69 )
            v49 = v69;
        }
LABEL_52:
        v118 = v50;
        if ( *v49 != -8 )
          --v119;
        *v49 = v22;
        v28 = 0;
        v49[1] = 0;
LABEL_21:
        v28 = (_QWORD **)**v28;
        goto LABEL_22;
      }
LABEL_20:
      v27 = v25[1];
      v28 = (_QWORD **)(v27 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v27 & 4) != 0 || !v28 )
        goto LABEL_21;
LABEL_22:
      v29 = sub_157ED20((__int64)v28);
      v30 = *(unsigned __int8 *)(v29 + 16);
      if ( (unsigned int)(v30 - 73) <= 1 )
      {
        if ( (_BYTE)v30 == 74 )
        {
          v4 = *(_QWORD *)(v29 - 24);
          v5 = 0;
          if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
          {
            if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
              v6 = *(_QWORD *)(v4 - 8);
            else
              v6 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
            v5 = *(_QWORD *)(v6 + 24);
          }
        }
        else
        {
          v5 = sub_1F5FF70(v29);
        }
        v7 = *(_QWORD *)(v23 - 24);
        if ( v5 != v7 )
          goto LABEL_9;
      }
      else
      {
        v7 = *(_QWORD *)(v23 - 24);
        v29 = 0;
        if ( v7 )
          goto LABEL_9;
      }
      v31 = *(unsigned int *)(a2 + 56);
      if ( !(_DWORD)v31 )
        goto LABEL_9;
      v32 = *(_QWORD *)(a2 + 40);
      v33 = (v31 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v34 = (__int64 *)(v32 + 16LL * v33);
      v35 = *v34;
      if ( v29 == *v34 )
      {
LABEL_26:
        if ( v34 == (__int64 *)(v32 + 16 * v31) )
          goto LABEL_9;
        v36 = *((_DWORD *)v34 + 2);
        if ( v36 == -1 )
          goto LABEL_9;
        v37 = *(_DWORD *)(a2 + 88);
        if ( !v37 )
        {
          ++*(_QWORD *)(a2 + 64);
          goto LABEL_122;
        }
        v38 = *(_QWORD *)(a2 + 72);
        v39 = (v37 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v40 = (unsigned __int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( *v40 != v23 )
        {
          v102 = 1;
          v103 = 0;
          while ( v41 != -8 )
          {
            if ( !v103 && v41 == -16 )
              v103 = v40;
            v39 = (v37 - 1) & (v102 + v39);
            v40 = (unsigned __int64 *)(v38 + 16LL * v39);
            v41 = *v40;
            if ( *v40 == v23 )
              goto LABEL_30;
            ++v102;
          }
          v104 = *(_DWORD *)(a2 + 80);
          if ( v103 )
            v40 = v103;
          ++*(_QWORD *)(a2 + 64);
          v100 = v104 + 1;
          if ( 4 * v100 < 3 * v37 )
          {
            if ( v37 - *(_DWORD *)(a2 + 84) - v100 <= v37 >> 3 )
            {
              sub_1F61760(a2 + 64, v37);
              v105 = *(_DWORD *)(a2 + 88);
              if ( !v105 )
              {
LABEL_199:
                ++*(_DWORD *)(a2 + 80);
                BUG();
              }
              v106 = v105 - 1;
              v107 = *(_QWORD *)(a2 + 72);
              v108 = 0;
              LODWORD(v109) = v106 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v110 = 1;
              v100 = *(_DWORD *)(a2 + 80) + 1;
              v40 = (unsigned __int64 *)(v107 + 16LL * (unsigned int)v109);
              v111 = *v40;
              if ( v23 != *v40 )
              {
                while ( v111 != -8 )
                {
                  if ( v111 == -16 && !v108 )
                    v108 = v40;
                  v109 = v106 & (unsigned int)(v109 + v110);
                  v40 = (unsigned __int64 *)(v107 + 16 * v109);
                  v111 = *v40;
                  if ( *v40 == v23 )
                    goto LABEL_124;
                  ++v110;
                }
                if ( v108 )
                  v40 = v108;
              }
            }
            goto LABEL_124;
          }
LABEL_122:
          sub_1F61760(a2 + 64, 2 * v37);
          v96 = *(_DWORD *)(a2 + 88);
          if ( !v96 )
            goto LABEL_199;
          v97 = v96 - 1;
          v98 = *(_QWORD *)(a2 + 72);
          LODWORD(v99) = v97 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v100 = *(_DWORD *)(a2 + 80) + 1;
          v40 = (unsigned __int64 *)(v98 + 16LL * (unsigned int)v99);
          v101 = *v40;
          if ( *v40 != v23 )
          {
            v112 = 1;
            v113 = 0;
            while ( v101 != -8 )
            {
              if ( !v113 && v101 == -16 )
                v113 = v40;
              v99 = v97 & (unsigned int)(v99 + v112);
              v40 = (unsigned __int64 *)(v98 + 16 * v99);
              v101 = *v40;
              if ( *v40 == v23 )
                goto LABEL_124;
              ++v112;
            }
            if ( v113 )
              v40 = v113;
          }
LABEL_124:
          *(_DWORD *)(a2 + 80) = v100;
          if ( *v40 != -8 )
            --*(_DWORD *)(a2 + 84);
          *v40 = v23;
          *((_DWORD *)v40 + 2) = 0;
        }
LABEL_30:
        *((_DWORD *)v40 + 2) = v36;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v115 == v3 )
          goto LABEL_31;
      }
      else
      {
        v94 = 1;
        while ( v35 != -8 )
        {
          v95 = v94 + 1;
          v33 = (v31 - 1) & (v94 + v33);
          v34 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v29 == *v34 )
            goto LABEL_26;
          v94 = v95;
        }
LABEL_9:
        v8 = sub_157ED20(v7);
        v9 = *(_DWORD *)(a2 + 24);
        v10 = v8;
        if ( v9 )
        {
          v11 = *(_QWORD *)(a2 + 8);
          v12 = ((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9);
          v13 = (v9 - 1) & v12;
          v14 = (__int64 *)(v11 + 16LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
          {
LABEL_11:
            v16 = *(_DWORD *)(a2 + 88);
            v17 = a2 + 64;
            if ( !v16 )
              goto LABEL_73;
            goto LABEL_12;
          }
          v55 = 1;
          v56 = 0;
          while ( v15 != -8 )
          {
            if ( !v56 && v15 == -16 )
              v56 = v14;
            v13 = (v9 - 1) & (v55 + v13);
            v14 = (__int64 *)(v11 + 16LL * v13);
            v15 = *v14;
            if ( v10 == *v14 )
              goto LABEL_11;
            ++v55;
          }
          v57 = *(_DWORD *)(a2 + 16);
          if ( v56 )
            v14 = v56;
          ++*(_QWORD *)a2;
          v58 = v57 + 1;
          if ( 4 * v58 < 3 * v9 )
          {
            if ( v9 - *(_DWORD *)(a2 + 20) - v58 <= v9 >> 3 )
            {
              v114 = v12;
              sub_1F61920(a2, v9);
              v88 = *(_DWORD *)(a2 + 24);
              if ( !v88 )
              {
LABEL_202:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v89 = v88 - 1;
              v90 = *(_QWORD *)(a2 + 8);
              v80 = 0;
              v91 = 1;
              LODWORD(v92) = v89 & v114;
              v58 = *(_DWORD *)(a2 + 16) + 1;
              v14 = (__int64 *)(v90 + 16LL * (v89 & v114));
              v93 = *v14;
              if ( v10 != *v14 )
              {
                while ( v93 != -8 )
                {
                  if ( v93 == -16 && !v80 )
                    v80 = v14;
                  v92 = v89 & (unsigned int)(v92 + v91);
                  v14 = (__int64 *)(v90 + 16 * v92);
                  v93 = *v14;
                  if ( v10 == *v14 )
                    goto LABEL_70;
                  ++v91;
                }
LABEL_100:
                if ( v80 )
                  v14 = v80;
                goto LABEL_70;
              }
            }
            goto LABEL_70;
          }
        }
        else
        {
          ++*(_QWORD *)a2;
        }
        sub_1F61920(a2, 2 * v9);
        v74 = *(_DWORD *)(a2 + 24);
        if ( !v74 )
          goto LABEL_202;
        v75 = v74 - 1;
        v76 = *(_QWORD *)(a2 + 8);
        LODWORD(v77) = (v74 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v58 = *(_DWORD *)(a2 + 16) + 1;
        v14 = (__int64 *)(v76 + 16LL * (unsigned int)v77);
        v78 = *v14;
        if ( v10 != *v14 )
        {
          v79 = 1;
          v80 = 0;
          while ( v78 != -8 )
          {
            if ( !v80 && v78 == -16 )
              v80 = v14;
            v77 = v75 & (unsigned int)(v77 + v79);
            v14 = (__int64 *)(v76 + 16 * v77);
            v78 = *v14;
            if ( v10 == *v14 )
              goto LABEL_70;
            ++v79;
          }
          goto LABEL_100;
        }
LABEL_70:
        *(_DWORD *)(a2 + 16) = v58;
        if ( *v14 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v14 = v10;
        v17 = a2 + 64;
        *((_DWORD *)v14 + 2) = 0;
        v16 = *(_DWORD *)(a2 + 88);
        if ( !v16 )
        {
LABEL_73:
          ++*(_QWORD *)(a2 + 64);
          goto LABEL_74;
        }
LABEL_12:
        v18 = *(_QWORD *)(a2 + 72);
        v19 = (v16 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v20 = (unsigned __int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( *v20 != v23 )
        {
          v51 = 1;
          v52 = 0;
          while ( v21 != -8 )
          {
            if ( !v52 && v21 == -16 )
              v52 = v20;
            v19 = (v16 - 1) & (v51 + v19);
            v20 = (unsigned __int64 *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( *v20 == v23 )
              goto LABEL_13;
            ++v51;
          }
          v53 = *(_DWORD *)(a2 + 80);
          if ( v52 )
            v20 = v52;
          ++*(_QWORD *)(a2 + 64);
          v54 = v53 + 1;
          if ( 4 * v54 >= 3 * v16 )
          {
LABEL_74:
            sub_1F61760(v17, 2 * v16);
            v59 = *(_DWORD *)(a2 + 88);
            if ( !v59 )
              goto LABEL_201;
            v60 = v59 - 1;
            v61 = *(_QWORD *)(a2 + 72);
            LODWORD(v62) = (v59 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v54 = *(_DWORD *)(a2 + 80) + 1;
            v20 = (unsigned __int64 *)(v61 + 16LL * (unsigned int)v62);
            v63 = *v20;
            if ( *v20 != v23 )
            {
              v64 = 1;
              v65 = 0;
              while ( v63 != -8 )
              {
                if ( !v65 && v63 == -16 )
                  v65 = v20;
                v62 = v60 & (unsigned int)(v62 + v64);
                v20 = (unsigned __int64 *)(v61 + 16 * v62);
                v63 = *v20;
                if ( *v20 == v23 )
                  goto LABEL_61;
                ++v64;
              }
              if ( v65 )
                v20 = v65;
            }
          }
          else if ( v16 - *(_DWORD *)(a2 + 84) - v54 <= v16 >> 3 )
          {
            sub_1F61760(v17, v16);
            v81 = *(_DWORD *)(a2 + 88);
            if ( !v81 )
            {
LABEL_201:
              ++*(_DWORD *)(a2 + 80);
              BUG();
            }
            v82 = v81 - 1;
            v83 = *(_QWORD *)(a2 + 72);
            v84 = 0;
            LODWORD(v85) = (v81 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v86 = 1;
            v54 = *(_DWORD *)(a2 + 80) + 1;
            v20 = (unsigned __int64 *)(v83 + 16LL * (unsigned int)v85);
            v87 = *v20;
            if ( *v20 != v23 )
            {
              while ( v87 != -8 )
              {
                if ( v87 == -16 && !v84 )
                  v84 = v20;
                v85 = v82 & (unsigned int)(v85 + v86);
                v20 = (unsigned __int64 *)(v83 + 16 * v85);
                v87 = *v20;
                if ( *v20 == v23 )
                  goto LABEL_61;
                ++v86;
              }
              if ( v84 )
                v20 = v84;
            }
          }
LABEL_61:
          *(_DWORD *)(a2 + 80) = v54;
          if ( *v20 != -8 )
            --*(_DWORD *)(a2 + 84);
          *v20 = v23;
          *((_DWORD *)v20 + 2) = 0;
        }
LABEL_13:
        *((_DWORD *)v20 + 2) = *((_DWORD *)v14 + 2);
LABEL_14:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v115 == v3 )
          goto LABEL_31;
      }
    }
    ++v116;
    goto LABEL_82;
  }
LABEL_31:
  if ( v120 )
  {
    v42 = v117;
    v43 = &v117[2 * v120];
    do
    {
      if ( *v42 != -8 && *v42 != -16 )
      {
        v44 = v42[1];
        if ( (v44 & 4) != 0 )
        {
          v45 = (unsigned __int64 *)(v44 & 0xFFFFFFFFFFFFFFF8LL);
          v46 = v45;
          if ( v45 )
          {
            if ( (unsigned __int64 *)*v45 != v45 + 2 )
              _libc_free(*v45);
            j_j___libc_free_0(v46, 48);
          }
        }
      }
      v42 += 2;
    }
    while ( v43 != v42 );
  }
  return j___libc_free_0(v117);
}
