// Function: sub_2FB4A90
// Address: 0x2fb4a90
//
void __fastcall sub_2FB4A90(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r12
  __int64 *v6; // r13
  __int64 v7; // r8
  int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned __int64 v15; // rsi
  int v16; // r15d
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 *v20; // r12
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 *v23; // rbx
  __int64 v24; // r13
  __int64 *v25; // r15
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // r10
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  _QWORD *v34; // r8
  __int64 v35; // r9
  __int64 v36; // rsi
  int v37; // ecx
  unsigned int v38; // r8d
  int v39; // edx
  unsigned int v40; // eax
  __int64 v41; // rax
  __int64 *v42; // rdi
  int v43; // eax
  __int64 v44; // r8
  unsigned __int64 v45; // rcx
  int v46; // r10d
  unsigned int v47; // eax
  __int64 v48; // r15
  unsigned int v49; // eax
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rbx
  _QWORD *v54; // r12
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rdx
  _QWORD *v60; // r11
  unsigned int v61; // esi
  __int64 v62; // rdi
  __int64 *v63; // rcx
  __int64 v64; // r13
  unsigned __int64 v65; // rax
  _QWORD *v66; // rdx
  _QWORD *v67; // rdi
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdx
  _QWORD *v71; // r11
  __int64 v72; // rsi
  __int64 *v73; // rcx
  __int64 v74; // rax
  unsigned __int64 v75; // r13
  _QWORD *v76; // rdx
  _QWORD *v77; // rdi
  unsigned __int64 v78; // [rsp+8h] [rbp-398h]
  __int64 v79; // [rsp+18h] [rbp-388h]
  __int64 v80; // [rsp+20h] [rbp-380h]
  __int64 v81; // [rsp+20h] [rbp-380h]
  __int64 v82; // [rsp+28h] [rbp-378h]
  __int64 v83; // [rsp+28h] [rbp-378h]
  int v84; // [rsp+40h] [rbp-360h]
  __int64 v85; // [rsp+48h] [rbp-358h]
  __int64 v86; // [rsp+48h] [rbp-358h]
  int v87; // [rsp+48h] [rbp-358h]
  __int64 *v88; // [rsp+60h] [rbp-340h]
  __int64 v89; // [rsp+60h] [rbp-340h]
  __int64 v90; // [rsp+60h] [rbp-340h]
  __int64 v91; // [rsp+60h] [rbp-340h]
  _QWORD *v92; // [rsp+60h] [rbp-340h]
  __int64 v93; // [rsp+60h] [rbp-340h]
  __int64 v94; // [rsp+68h] [rbp-338h]
  _QWORD *v95; // [rsp+68h] [rbp-338h]
  __int64 *v96; // [rsp+68h] [rbp-338h]
  __int64 v97; // [rsp+68h] [rbp-338h]
  __int64 *v98; // [rsp+70h] [rbp-330h] BYREF
  __int64 v99; // [rsp+78h] [rbp-328h]
  _BYTE v100[32]; // [rsp+80h] [rbp-320h] BYREF
  _QWORD v101[5]; // [rsp+A0h] [rbp-300h] BYREF
  _BYTE *v102; // [rsp+C8h] [rbp-2D8h]
  __int64 v103; // [rsp+D0h] [rbp-2D0h]
  _BYTE v104[48]; // [rsp+D8h] [rbp-2C8h] BYREF
  int v105; // [rsp+108h] [rbp-298h]
  __int64 v106; // [rsp+110h] [rbp-290h]
  _QWORD *v107; // [rsp+118h] [rbp-288h]
  __int64 v108; // [rsp+120h] [rbp-280h]
  unsigned int v109; // [rsp+128h] [rbp-278h]
  _QWORD *v110; // [rsp+130h] [rbp-270h]
  __int64 v111; // [rsp+138h] [rbp-268h]
  _QWORD v112[3]; // [rsp+140h] [rbp-260h] BYREF
  _BYTE *v113; // [rsp+158h] [rbp-248h]
  __int64 v114; // [rsp+160h] [rbp-240h]
  _BYTE v115[568]; // [rsp+168h] [rbp-238h] BYREF

  v2 = a1;
  v85 = a1 + 192;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v80 = v3;
  v4 = *(__int64 **)(v3 + 64);
  v88 = &v4[*(unsigned int *)(v3 + 72)];
  if ( v4 != v88 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(*v4 + 8);
      v94 = *v4;
      if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v11 & 6) == 0 )
        break;
LABEL_8:
      if ( v88 == ++v4 )
      {
        v2 = a1;
        goto LABEL_16;
      }
    }
    v12 = sub_2FB3BF0(v85, v11, 0);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = v94;
    v84 = v12;
    v15 = *(unsigned int *)(v13 + 160);
    v16 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                    + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + v12));
    v17 = v16 & 0x7FFFFFFF;
    v5 = 8LL * (v16 & 0x7FFFFFFF);
    if ( (v16 & 0x7FFFFFFFu) < (unsigned int)v15 )
    {
      v5 = 8LL * v17;
      v6 = *(__int64 **)(*(_QWORD *)(v13 + 152) + v5);
      if ( v6 )
      {
LABEL_4:
        v7 = *(_QWORD *)(v14 + 8);
        v8 = *(_DWORD *)(a1 + 84);
        v9 = *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v9 )
        {
          v10 = *(_QWORD *)(v9 + 24);
        }
        else
        {
          v69 = *(_QWORD *)(v13 + 32);
          v70 = *(unsigned int *)(v69 + 304);
          v71 = *(_QWORD **)(v69 + 296);
          if ( *(_DWORD *)(v69 + 304) )
          {
            do
            {
              while ( 1 )
              {
                v72 = v70 >> 1;
                v73 = &v71[2 * (v70 >> 1)];
                if ( (*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7 >> 1) & 3) < (*(_DWORD *)((*v73 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v73 >> 1) & 3) )
                  break;
                v71 = v73 + 2;
                v70 = v70 - v72 - 1;
                if ( v70 <= 0 )
                  goto LABEL_90;
              }
              v70 >>= 1;
            }
            while ( v72 > 0 );
          }
LABEL_90:
          v10 = *(v71 - 1);
        }
        if ( !(unsigned __int8)sub_2FB04E0(v7, (__int64)v6) )
          sub_2FB2810(
            a1,
            v10,
            (_QWORD *)(a1 + 712LL * ((v84 != 0) & (unsigned __int8)(v8 != 0)) + 424),
            v6,
            -1,
            -1,
            0,
            0);
        goto LABEL_8;
      }
    }
    v18 = v17 + 1;
    if ( (unsigned int)v15 < v18 )
    {
      v68 = v18;
      if ( v18 != v15 )
      {
        if ( v18 >= v15 )
        {
          v74 = *(_QWORD *)(v13 + 168);
          v75 = v68 - v15;
          if ( v68 > *(unsigned int *)(v13 + 164) )
          {
            v79 = *(_QWORD *)(v13 + 168);
            v83 = v94;
            v97 = *(_QWORD *)(a1 + 8);
            sub_C8D5F0(v13 + 152, (const void *)(v13 + 168), v68, 8u, v14, v13);
            v13 = v97;
            v74 = v79;
            v14 = v83;
            v15 = *(unsigned int *)(v97 + 160);
          }
          v19 = *(_QWORD *)(v13 + 152);
          v76 = (_QWORD *)(v19 + 8 * v15);
          v77 = &v76[v75];
          if ( v76 != v77 )
          {
            do
              *v76++ = v74;
            while ( v77 != v76 );
            LODWORD(v15) = *(_DWORD *)(v13 + 160);
            v19 = *(_QWORD *)(v13 + 152);
          }
          *(_DWORD *)(v13 + 160) = v75 + v15;
          goto LABEL_14;
        }
        *(_DWORD *)(v13 + 160) = v18;
      }
    }
    v19 = *(_QWORD *)(v13 + 152);
LABEL_14:
    v82 = v14;
    v20 = (__int64 *)(v19 + v5);
    v95 = (_QWORD *)v13;
    v21 = sub_2E10F30(v16);
    *v20 = v21;
    v6 = (__int64 *)v21;
    sub_2E11E80(v95, v21);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = v82;
    goto LABEL_4;
  }
LABEL_16:
  v105 = 0;
  v98 = (__int64 *)v100;
  v99 = 0x400000000LL;
  v102 = v104;
  v103 = 0x600000000LL;
  v110 = v112;
  v113 = v115;
  v114 = 0x1000000000LL;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v111 = 0;
  v112[0] = 0;
  v112[1] = 0;
  v22 = *(_QWORD *)(v80 + 104);
  memset(v101, 0, sizeof(v101));
  if ( !v22 )
    goto LABEL_44;
  do
  {
    v23 = *(__int64 **)(v22 + 64);
    v96 = &v23[*(unsigned int *)(v22 + 72)];
    if ( v96 != v23 )
    {
      while ( 1 )
      {
        v35 = *v23;
        v36 = *(_QWORD *)(*v23 + 8);
        if ( (v36 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v36 & 6) == 0 )
          break;
LABEL_24:
        if ( v96 == ++v23 )
          goto LABEL_39;
      }
      v37 = *(_DWORD *)(v2 + 380);
      if ( !v37 )
        goto LABEL_38;
      v38 = *(_DWORD *)((*(_QWORD *)(*v23 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v39 = *(_DWORD *)(v2 + 376);
      v40 = *(_DWORD *)((*(_QWORD *)(v2 + 192) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v2 + 192) >> 1) & 3;
      if ( v39 )
      {
        if ( v40 > v38 )
          goto LABEL_38;
        v41 = *(_QWORD *)(v2 + 8LL * (unsigned int)(v37 - 1) + 288);
      }
      else
      {
        if ( v40 > v38 )
          goto LABEL_38;
        v41 = *(_QWORD *)(v2 + 16LL * (unsigned int)(v37 - 1) + 200);
      }
      if ( (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) > v38 )
      {
        v42 = (__int64 *)(v2 + 192);
        if ( v39 )
        {
          v91 = *v23;
          v43 = sub_2FB3A50((__int64)v42, v36, 0);
          v35 = v91;
          v39 = v43;
        }
        else
        {
          if ( (*(_DWORD *)((*(_QWORD *)(v2 + 200) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(v2 + 200) >> 1) & 3) > v38 )
          {
            v58 = 0;
          }
          else
          {
            LODWORD(v58) = 0;
            do
              v58 = (unsigned int)(v58 + 1);
            while ( (*(_DWORD *)((v42[2 * v58 + 1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                   | (unsigned int)(v42[2 * v58 + 1] >> 1) & 3) <= v38 );
            v42 += 2 * v58;
          }
          if ( (*(_DWORD *)((*v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v42 >> 1) & 3) <= v38 )
            v39 = *(_DWORD *)(v2 + 4 * v58 + 336);
        }
LABEL_34:
        v44 = *(_QWORD *)(v2 + 8);
        v45 = *(unsigned int *)(v44 + 160);
        v46 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v2 + 72) + 16LL)
                        + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v2 + 72) + 64LL) + v39));
        v47 = v46 & 0x7FFFFFFF;
        v48 = 8LL * (v46 & 0x7FFFFFFF);
        if ( (v46 & 0x7FFFFFFFu) < (unsigned int)v45 )
        {
          v24 = *(_QWORD *)(*(_QWORD *)(v44 + 152) + 8LL * v47);
          if ( v24 )
          {
LABEL_20:
            v89 = v35;
            v25 = sub_2FB2150(*(_QWORD *)(v22 + 112), *(_QWORD *)(v22 + 120), v24);
            if ( !(unsigned __int8)sub_2FB04E0(*(_QWORD *)(v89 + 8), (__int64)v25) )
            {
              v26 = *(_QWORD *)(v2 + 8);
              v27 = *(_QWORD *)(v26 + 32);
              v28 = *(_QWORD *)(v89 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              v29 = *(_QWORD *)(v28 + 16);
              if ( v29 )
              {
                v30 = *(_QWORD *)(v29 + 24);
              }
              else
              {
                v59 = *(unsigned int *)(v27 + 304);
                v60 = *(_QWORD **)(v27 + 296);
                if ( *(_DWORD *)(v27 + 304) )
                {
                  v61 = *(_DWORD *)(v28 + 24) | (*(__int64 *)(v89 + 8) >> 1) & 3;
                  do
                  {
                    while ( 1 )
                    {
                      v62 = v59 >> 1;
                      v63 = &v60[2 * (v59 >> 1)];
                      if ( v61 < (*(_DWORD *)((*v63 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v63 >> 1) & 3) )
                        break;
                      v60 = v63 + 2;
                      v59 = v59 - v62 - 1;
                      if ( v59 <= 0 )
                        goto LABEL_75;
                    }
                    v59 >>= 1;
                  }
                  while ( v62 > 0 );
                }
LABEL_75:
                v30 = *(v60 - 1);
              }
              v90 = v30;
              sub_2E1DCC0(
                (__int64)v101,
                *(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL),
                *(_QWORD *)(v26 + 32),
                *(_QWORD *)(v2 + 32),
                v26 + 56,
                v27);
              v31 = *(_QWORD *)(v22 + 112);
              v32 = *(_QWORD *)(v22 + 120);
              v33 = *(_QWORD *)(v2 + 8);
              v34 = *(_QWORD **)(v2 + 24);
              LODWORD(v99) = 0;
              sub_2E0B070(v24, (__int64)&v98, v31, v32, v34, *(_QWORD *)(v33 + 32));
              sub_2FB2810(v2, v90, v101, v25, *(_QWORD *)(v22 + 112), *(_QWORD *)(v22 + 120), v98, (unsigned int)v99);
            }
            goto LABEL_24;
          }
        }
        v49 = v47 + 1;
        if ( (unsigned int)v45 < v49 )
        {
          v57 = v49;
          if ( v49 != v45 )
          {
            if ( v49 >= v45 )
            {
              v64 = *(_QWORD *)(v44 + 168);
              v65 = v49 - v45;
              if ( v57 > *(unsigned int *)(v44 + 164) )
              {
                v78 = v65;
                v81 = v35;
                v87 = v46;
                v93 = *(_QWORD *)(v2 + 8);
                sub_C8D5F0(v44 + 152, (const void *)(v44 + 168), v57, 8u, v44, v35);
                v44 = v93;
                v65 = v78;
                v35 = v81;
                v46 = v87;
                v45 = *(unsigned int *)(v93 + 160);
              }
              v50 = *(_QWORD *)(v44 + 152);
              v66 = (_QWORD *)(v50 + 8 * v45);
              v67 = &v66[v65];
              if ( v66 != v67 )
              {
                do
                  *v66++ = v64;
                while ( v67 != v66 );
                LODWORD(v45) = *(_DWORD *)(v44 + 160);
                v50 = *(_QWORD *)(v44 + 152);
              }
              *(_DWORD *)(v44 + 160) = v65 + v45;
              goto LABEL_37;
            }
            *(_DWORD *)(v44 + 160) = v49;
          }
        }
        v50 = *(_QWORD *)(v44 + 152);
LABEL_37:
        v86 = v35;
        v92 = (_QWORD *)v44;
        v51 = sub_2E10F30(v46);
        *(_QWORD *)(v50 + v48) = v51;
        v24 = v51;
        sub_2E11E80(v92, v51);
        v35 = v86;
        goto LABEL_20;
      }
LABEL_38:
      v39 = 0;
      goto LABEL_34;
    }
LABEL_39:
    v22 = *(_QWORD *)(v22 + 104);
  }
  while ( v22 );
  if ( v113 != v115 )
    _libc_free((unsigned __int64)v113);
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
LABEL_44:
  v52 = v109;
  if ( v109 )
  {
    v53 = v107;
    v54 = &v107[19 * v109];
    do
    {
      if ( *v53 != -4096 && *v53 != -8192 )
      {
        v55 = v53[10];
        if ( (_QWORD *)v55 != v53 + 12 )
          _libc_free(v55);
        v56 = v53[1];
        if ( (_QWORD *)v56 != v53 + 3 )
          _libc_free(v56);
      }
      v53 += 19;
    }
    while ( v54 != v53 );
    v52 = v109;
  }
  sub_C7D6A0((__int64)v107, 152 * v52, 8);
  if ( v102 != v104 )
    _libc_free((unsigned __int64)v102);
  if ( v98 != (__int64 *)v100 )
    _libc_free((unsigned __int64)v98);
}
