// Function: sub_1789990
// Address: 0x1789990
//
__int64 __fastcall sub_1789990(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rbx
  _BYTE *v9; // r10
  int v10; // ecx
  __int64 v11; // rdx
  __int64 *v12; // r13
  unsigned __int64 v13; // r14
  __int64 *v14; // rax
  int v15; // eax
  __int64 v16; // rax
  char v17; // r15
  char v18; // r14
  unsigned int v19; // r13d
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v25; // rax
  bool v26; // al
  __int64 v27; // r9
  bool v28; // cl
  unsigned int v29; // eax
  __int64 v30; // r8
  __int64 i; // rdx
  __int64 v32; // rax
  __int64 v33; // r13
  _BYTE *v34; // rdi
  int v35; // r15d
  __int64 v36; // r15
  __int64 *v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r14
  __int64 v42; // rsi
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // esi
  __int64 v50; // rsi
  __int64 **v51; // rdx
  __int64 *v52; // rdi
  unsigned __int64 v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // r14
  unsigned int j; // ebx
  __int64 v59; // r11
  __int64 *v60; // r13
  __int64 v61; // r15
  _QWORD *v62; // rax
  _BYTE *v63; // r10
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 *v66; // rax
  __int64 *v67; // rax
  _BYTE *v68; // r10
  __int64 *v69; // rdi
  __int64 *v70; // rcx
  __int64 *v71; // rax
  __int64 v72; // rdx
  __int64 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r8
  __int64 v76; // r14
  __int64 v77; // rcx
  __int64 v78; // r9
  __int64 v79; // r13
  __int64 v80; // rbx
  __int64 v81; // rdx
  _QWORD *v82; // rax
  __int64 v83; // rsi
  unsigned __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // r15
  __int64 v88; // rdx
  __int64 v89; // r12
  int v90; // eax
  __int64 v91; // rax
  int v92; // edx
  char v93; // al
  __int64 v94; // rax
  __int64 *v95; // rax
  _BYTE *v96; // [rsp+8h] [rbp-1C8h]
  __int64 v97; // [rsp+18h] [rbp-1B8h]
  _BYTE *v98; // [rsp+18h] [rbp-1B8h]
  __int64 v99; // [rsp+20h] [rbp-1B0h]
  int v100; // [rsp+20h] [rbp-1B0h]
  _BYTE *v101; // [rsp+20h] [rbp-1B0h]
  int v102; // [rsp+20h] [rbp-1B0h]
  __int64 v103; // [rsp+20h] [rbp-1B0h]
  unsigned int v104; // [rsp+28h] [rbp-1A8h]
  __int64 v105; // [rsp+28h] [rbp-1A8h]
  _BYTE *v106; // [rsp+28h] [rbp-1A8h]
  _BYTE *v107; // [rsp+30h] [rbp-1A0h]
  __int64 v108; // [rsp+30h] [rbp-1A0h]
  unsigned int v109; // [rsp+30h] [rbp-1A0h]
  bool v110; // [rsp+30h] [rbp-1A0h]
  _BYTE *v111; // [rsp+38h] [rbp-198h]
  __int64 v112; // [rsp+38h] [rbp-198h]
  __int64 v113; // [rsp+38h] [rbp-198h]
  _BYTE *v114; // [rsp+38h] [rbp-198h]
  char v115; // [rsp+40h] [rbp-190h]
  _BYTE *v116; // [rsp+40h] [rbp-190h]
  _BYTE *v117; // [rsp+40h] [rbp-190h]
  _QWORD v119[2]; // [rsp+50h] [rbp-180h] BYREF
  __int64 v120[2]; // [rsp+60h] [rbp-170h] BYREF
  __int16 v121; // [rsp+70h] [rbp-160h]
  __int64 *v122; // [rsp+80h] [rbp-150h] BYREF
  __int64 v123; // [rsp+88h] [rbp-148h]
  _BYTE v124[128]; // [rsp+90h] [rbp-140h] BYREF
  _BYTE *v125; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v126; // [rsp+118h] [rbp-B8h]
  _BYTE s[176]; // [rsp+120h] [rbp-B0h] BYREF

  v6 = a2;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *v7;
  v9 = v124;
  v10 = 0;
  v122 = (__int64 *)v124;
  v11 = 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
  v123 = 0x1000000000LL;
  v12 = (__int64 *)(v8 - v11);
  v13 = 0xAAAAAAAAAAAAAAABLL * (v11 >> 3);
  v14 = (__int64 *)v124;
  if ( (unsigned __int64)v11 > 0x180 )
  {
    sub_16CD150((__int64)&v122, v124, 0xAAAAAAAAAAAAAAABLL * (v11 >> 3), 8, a5, a6);
    v10 = v123;
    v9 = v124;
    v14 = &v122[(unsigned int)v123];
  }
  if ( v12 != (__int64 *)v8 )
  {
    do
    {
      if ( v14 )
        *v14 = *v12;
      v12 += 3;
      ++v14;
    }
    while ( v12 != (__int64 *)v8 );
    v10 = v123;
  }
  v15 = *(_DWORD *)(a2 + 20);
  LODWORD(v123) = v10 + v13;
  v16 = v15 & 0xFFFFFFF;
  if ( (_DWORD)v16 != 1 )
  {
    v115 = 1;
    v17 = 0;
    v18 = 1;
    v19 = 1;
    while ( 1 )
    {
      v20 = (*(_BYTE *)(v6 + 23) & 0x40) != 0 ? *(_QWORD *)(v6 - 8) : v6 - 24 * v16;
      v21 = *(_QWORD *)(v20 + 24LL * v19);
      if ( *(_BYTE *)(v21 + 16) != 56 )
        break;
      v25 = *(_QWORD *)(v21 + 8);
      if ( !v25
        || *(_QWORD *)(v25 + 8)
        || *(_QWORD *)v21 != *(_QWORD *)v8
        || (*(_DWORD *)(v21 + 20) & 0xFFFFFFF) != (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) )
      {
        break;
      }
      v107 = v9;
      v26 = sub_15FA300(v21);
      v9 = v107;
      v28 = v26;
      if ( v18 )
      {
        v18 = 0;
        if ( *(_BYTE *)(*(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)) + 16LL) == 53 )
        {
          v106 = v107;
          v110 = v26;
          v93 = sub_15FA290(v21);
          v9 = v106;
          v28 = v110;
          v18 = v93;
        }
      }
      v29 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      if ( v29 )
      {
        v30 = v29 - 1;
        for ( i = 0; ; ++i )
        {
          v27 = *(_QWORD *)(v8 + 24 * (i - v29));
          v32 = *(_QWORD *)(v21 + 24 * (i - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)));
          if ( v32 != v27 )
          {
            v22 = (unsigned __int64)v122;
            if ( *(_BYTE *)(v27 + 16) == 13 || *(_BYTE *)(v32 + 16) == 13 || *(_QWORD *)v32 != *(_QWORD *)v27 || v17 )
              goto LABEL_16;
            v122[i] = 0;
            v17 = 1;
          }
          if ( v30 == i )
            break;
          v29 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
        }
      }
      ++v19;
      v115 &= v28;
      v16 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v16 == v19 )
      {
        if ( v18 )
          break;
        v33 = (unsigned int)v123;
        v34 = s;
        v125 = s;
        v35 = v123;
        v126 = 0x1000000000LL;
        if ( (unsigned int)v123 > 0x10 )
        {
          v21 = (__int64)s;
          v114 = v9;
          sub_16CD150((__int64)&v125, s, (unsigned int)v123, 8, (int)&v125, v27);
          v34 = v125;
          v9 = v114;
        }
        LODWORD(v126) = v35;
        if ( 8 * v33 )
        {
          v21 = 0;
          v111 = v9;
          memset(v34, 0, 8 * v33);
          v9 = v111;
        }
        v104 = v123;
        if ( (_DWORD)v123 )
        {
          v97 = v8;
          v36 = 0;
          v96 = v9;
          do
          {
            if ( !v122[v36] )
            {
              v37 = *(__int64 **)(v97 + 24 * (v36 - (*(_DWORD *)(v97 + 20) & 0xFFFFFFF)));
              v119[0] = sub_1649960((__int64)v37);
              v119[1] = v38;
              v120[0] = (__int64)v119;
              v121 = 773;
              v120[1] = (__int64)".pn";
              v99 = *v37;
              v39 = sub_1648B60(64);
              v40 = v39;
              if ( v39 )
              {
                v41 = v39;
                sub_15F1EA0(v39, v99, 53, 0, 0, 0);
                *(_DWORD *)(v40 + 56) = v104;
                sub_164B780(v40, v120);
                sub_1648880(v40, *(_DWORD *)(v40 + 56), 1);
              }
              else
              {
                v41 = 0;
              }
              sub_157E9D0(*(_QWORD *)(v6 + 40) + 40LL, v40);
              v42 = *(_QWORD *)(v6 + 24);
              *(_QWORD *)(v40 + 32) = v6 + 24;
              v42 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v40 + 24) = v42 | *(_QWORD *)(v40 + 24) & 7LL;
              *(_QWORD *)(v42 + 8) = v40 + 24;
              *(_QWORD *)(v6 + 24) = *(_QWORD *)(v6 + 24) & 7LL | (v40 + 24);
              sub_170B990(*a1, v40);
              if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
                v45 = *(_QWORD *)(v6 - 8);
              else
                v45 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
              v46 = *(_QWORD *)(v45 + 24LL * *(unsigned int *)(v6 + 56) + 8);
              v47 = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
              if ( (_DWORD)v47 == *(_DWORD *)(v40 + 56) )
              {
                v103 = v46;
                sub_15F55D0(v40, v40, v47, v46, v43, v44);
                v46 = v103;
                LODWORD(v47) = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
              }
              v48 = ((_DWORD)v47 + 1) & 0xFFFFFFF;
              v49 = v48 | *(_DWORD *)(v40 + 20) & 0xF0000000;
              *(_DWORD *)(v40 + 20) = v49;
              if ( (v49 & 0x40000000) != 0 )
                v50 = *(_QWORD *)(v40 - 8);
              else
                v50 = v41 - 24 * v48;
              v51 = (__int64 **)(v50 + 24LL * (unsigned int)(v48 - 1));
              if ( *v51 )
              {
                v52 = v51[1];
                v53 = (unsigned __int64)v51[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v53 = v52;
                if ( v52 )
                  v52[2] = v52[2] & 3 | v53;
              }
              *v51 = v37;
              v54 = v37[1];
              v51[1] = (__int64 *)v54;
              if ( v54 )
                *(_QWORD *)(v54 + 16) = (unsigned __int64)(v51 + 1) | *(_QWORD *)(v54 + 16) & 3LL;
              v51[2] = (__int64 *)((unsigned __int64)(v37 + 1) | (unsigned __int64)v51[2] & 3);
              v37[1] = (__int64)v51;
              v55 = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
              v56 = (unsigned int)(v55 - 1);
              if ( (*(_BYTE *)(v40 + 23) & 0x40) != 0 )
                v57 = *(_QWORD *)(v40 - 8);
              else
                v57 = v41 - 24 * v55;
              v21 = 3LL * *(unsigned int *)(v40 + 56);
              *(_QWORD *)(v57 + 8 * v56 + 24LL * *(unsigned int *)(v40 + 56) + 8) = v46;
              v18 = 1;
              *(_QWORD *)&v125[8 * v36] = v40;
              v122[v36] = v40;
            }
            ++v36;
          }
          while ( v104 != v36 );
          v8 = v97;
          v9 = v96;
          if ( v18 )
          {
            v102 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
            if ( v102 != 1 )
            {
              v59 = v6;
              for ( j = 1; j != v102; ++j )
              {
                if ( (*(_BYTE *)(v59 + 23) & 0x40) != 0 )
                  v74 = *(_QWORD *)(v59 - 8);
                else
                  v74 = v59 - 24LL * (*(_DWORD *)(v59 + 20) & 0xFFFFFFF);
                v75 = (unsigned int)v126;
                v76 = 0;
                v77 = 3LL * *(unsigned int *)(v59 + 56);
                v78 = *(_QWORD *)(v74 + 8LL * j + 24LL * *(unsigned int *)(v59 + 56) + 8);
                if ( (_DWORD)v126 )
                {
                  v109 = j;
                  v79 = *(_QWORD *)(v74 + 24LL * j);
                  v80 = *(_QWORD *)(v74 + 8LL * j + 24LL * *(unsigned int *)(v59 + 56) + 8);
                  v105 = v59;
                  do
                  {
                    v87 = *(_QWORD *)&v125[8 * v76];
                    if ( v87 )
                    {
                      v88 = v76 - (*(_DWORD *)(v79 + 20) & 0xFFFFFFF);
                      v89 = *(_QWORD *)(v79 + 24 * v88);
                      v90 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
                      if ( v90 == *(_DWORD *)(v87 + 56) )
                      {
                        v113 = v75;
                        sub_15F55D0(*(_QWORD *)&v125[8 * v76], v21, v88, v77, v75, v78);
                        v75 = v113;
                        v90 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
                      }
                      v91 = (v90 + 1) & 0xFFFFFFF;
                      v92 = v91 | *(_DWORD *)(v87 + 20) & 0xF0000000;
                      *(_DWORD *)(v87 + 20) = v92;
                      if ( (v92 & 0x40000000) != 0 )
                        v81 = *(_QWORD *)(v87 - 8);
                      else
                        v81 = v87 - 24 * v91;
                      v82 = (_QWORD *)(v81 + 24LL * (unsigned int)(v91 - 1));
                      if ( *v82 )
                      {
                        v83 = v82[1];
                        v84 = v82[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v84 = v83;
                        if ( v83 )
                          *(_QWORD *)(v83 + 16) = *(_QWORD *)(v83 + 16) & 3LL | v84;
                      }
                      *v82 = v89;
                      if ( v89 )
                      {
                        v85 = *(_QWORD *)(v89 + 8);
                        v82[1] = v85;
                        if ( v85 )
                          *(_QWORD *)(v85 + 16) = (unsigned __int64)(v82 + 1) | *(_QWORD *)(v85 + 16) & 3LL;
                        v82[2] = (v89 + 8) | v82[2] & 3LL;
                        *(_QWORD *)(v89 + 8) = v82;
                      }
                      v86 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
                      if ( (*(_BYTE *)(v87 + 23) & 0x40) != 0 )
                        v21 = *(_QWORD *)(v87 - 8);
                      else
                        v21 = v87 - 24 * v86;
                      *(_QWORD *)(v21 + 8LL * (unsigned int)(v86 - 1) + 24LL * *(unsigned int *)(v87 + 56) + 8) = v80;
                    }
                    ++v76;
                  }
                  while ( v75 != v76 );
                  j = v109;
                  v59 = v105;
                }
              }
              v8 = v97;
              v9 = v96;
              v6 = v59;
            }
          }
          v104 = v123;
        }
        v60 = v122 + 1;
        v112 = *v122;
        v121 = 257;
        v61 = *(_QWORD *)(v8 + 56);
        v108 = v104 - 1LL;
        if ( !v61 )
        {
          v94 = *(_QWORD *)v112;
          if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
            v94 = **(_QWORD **)(v94 + 16);
          v61 = *(_QWORD *)(v94 + 24);
        }
        v98 = v9;
        v62 = sub_1648A60(72, v104);
        v63 = v98;
        v23 = (__int64)v62;
        if ( v62 )
        {
          v64 = (__int64)&v62[-3 * v104];
          v65 = *(_QWORD *)v112;
          if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
            v65 = **(_QWORD **)(v65 + 16);
          v100 = *(_DWORD *)(v65 + 8) >> 8;
          v66 = (__int64 *)sub_15F9F50(v61, (__int64)v60, v108);
          v67 = (__int64 *)sub_1646BA0(v66, v100);
          v68 = v98;
          v69 = v67;
          if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
          {
            v95 = sub_16463B0(v67, *(_QWORD *)(*(_QWORD *)v112 + 32LL));
            v68 = v98;
            v69 = v95;
          }
          else
          {
            v70 = &v60[v108];
            if ( v60 != v70 )
            {
              v71 = v60;
              while ( 1 )
              {
                v72 = *(_QWORD *)*v71;
                if ( *(_BYTE *)(v72 + 8) == 16 )
                  break;
                if ( v70 == ++v71 )
                  goto LABEL_82;
              }
              v73 = sub_16463B0(v69, *(_QWORD *)(v72 + 32));
              v68 = v98;
              v69 = v73;
            }
          }
LABEL_82:
          v101 = v68;
          sub_15F1EA0(v23, (__int64)v69, 32, v64, v104, 0);
          *(_QWORD *)(v23 + 56) = v61;
          *(_QWORD *)(v23 + 64) = sub_15F9F50(v61, (__int64)v60, v108);
          sub_15F9CE0(v23, v112, v60, v108, (__int64)v120);
          v63 = v101;
        }
        if ( v115 )
        {
          v117 = v63;
          sub_15FA2E0(v23, 1);
          v63 = v117;
        }
        v116 = v63;
        sub_1789760((__int64)a1, v23, v6);
        v9 = v116;
        if ( v125 == s )
        {
          v22 = (unsigned __int64)v122;
        }
        else
        {
          _libc_free((unsigned __int64)v125);
          v22 = (unsigned __int64)v122;
          v9 = v116;
        }
        goto LABEL_17;
      }
    }
  }
  v22 = (unsigned __int64)v122;
LABEL_16:
  v23 = 0;
LABEL_17:
  if ( (_BYTE *)v22 != v9 )
    _libc_free(v22);
  return v23;
}
