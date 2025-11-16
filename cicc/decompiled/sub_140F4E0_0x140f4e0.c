// Function: sub_140F4E0
// Address: 0x140f4e0
//
__int64 __fastcall sub_140F4E0(__int64 a1, __int64 a2)
{
  int v4; // r14d
  __int64 v5; // r15
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rsi
  int v16; // r14d
  int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rdi
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // rcx
  unsigned int v30; // edx
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // r8
  int v49; // eax
  __int64 v50; // rax
  int v51; // edi
  __int64 v52; // rdi
  __int64 *v53; // rax
  __int64 v54; // r9
  unsigned __int64 v55; // rdi
  __int64 v56; // r9
  __int64 v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // rsi
  __int64 v60; // rsi
  int v61; // eax
  __int64 v62; // rax
  int v63; // edi
  __int64 v64; // rdi
  _QWORD *v65; // rax
  __int64 v66; // r8
  unsigned __int64 v67; // rdi
  __int64 v68; // r8
  __int64 v69; // rdx
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // rsi
  __int64 v85; // rdx
  __int64 v86; // rcx
  int v87; // r9d
  _QWORD *v88; // r11
  int v89; // eax
  int v90; // eax
  int v91; // ecx
  int v92; // ecx
  __int64 v93; // rdi
  unsigned int v94; // edx
  __int64 v95; // rsi
  int v96; // r9d
  _QWORD *v97; // r8
  int v98; // edx
  int v99; // edx
  __int64 v100; // rsi
  int v101; // r8d
  _QWORD *v102; // rdi
  __int64 v103; // r10
  __int64 v104; // rcx
  __int64 v105; // [rsp+0h] [rbp-C0h]
  __int64 v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+10h] [rbp-B0h]
  __int64 v108; // [rsp+18h] [rbp-A8h]
  __int64 v109; // [rsp+20h] [rbp-A0h]
  unsigned int v110; // [rsp+28h] [rbp-98h]
  __int64 v111; // [rsp+30h] [rbp-90h]
  __int64 v112; // [rsp+38h] [rbp-88h]
  __int64 v113; // [rsp+48h] [rbp-78h] BYREF
  _BYTE v114[16]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v115; // [rsp+60h] [rbp-60h]
  _QWORD v116[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v117; // [rsp+80h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 20);
  v5 = *(_QWORD *)(a1 + 96);
  v115 = 257;
  v117 = 257;
  v6 = v4 & 0xFFFFFFF;
  v7 = sub_1648B60(64);
  v8 = v7;
  if ( v7 )
  {
    v111 = v7;
    sub_15F1EA0(v7, v5, 53, 0, 0, 0);
    *(_DWORD *)(v8 + 56) = v6;
    sub_164B780(v8, v116);
    sub_1648880(v8, *(unsigned int *)(v8 + 56), 1);
  }
  else
  {
    v111 = 0;
  }
  v9 = *(_QWORD *)(a1 + 32);
  if ( v9 )
  {
    v10 = *(__int64 **)(a1 + 40);
    sub_157E9D0(v9 + 40, v8);
    v11 = *(_QWORD *)(v8 + 24);
    v12 = *v10;
    *(_QWORD *)(v8 + 32) = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v8 + 24;
    *v10 = *v10 & 7 | (v8 + 24);
  }
  sub_164B780(v111, v114);
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 )
  {
    v113 = *(_QWORD *)(a1 + 24);
    sub_1623A60(&v113, v13, 2);
    if ( *(_QWORD *)(v8 + 48) )
      sub_161E7C0(v8 + 48);
    v14 = v113;
    *(_QWORD *)(v8 + 48) = v113;
    if ( v14 )
      sub_1623210(&v113, v14, v8 + 48);
  }
  v15 = *(_QWORD *)(a1 + 96);
  v16 = *(_DWORD *)(a2 + 20);
  v115 = 257;
  v17 = v16 & 0xFFFFFFF;
  v117 = 257;
  v18 = sub_1648B60(64);
  v19 = v18;
  if ( v18 )
  {
    v112 = v18;
    sub_15F1EA0(v18, v15, 53, 0, 0, 0);
    *(_DWORD *)(v19 + 56) = v17;
    sub_164B780(v19, v116);
    sub_1648880(v19, *(unsigned int *)(v19 + 56), 1);
  }
  else
  {
    v112 = 0;
  }
  v20 = *(_QWORD *)(a1 + 32);
  if ( v20 )
  {
    v21 = *(__int64 **)(a1 + 40);
    sub_157E9D0(v20 + 40, v19);
    v22 = *(_QWORD *)(v19 + 24);
    v23 = *v21;
    *(_QWORD *)(v19 + 32) = v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v19 + 24) = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v19 + 24;
    *v21 = *v21 & 7 | (v19 + 24);
  }
  sub_164B780(v112, v114);
  v24 = *(_QWORD *)(a1 + 24);
  if ( v24 )
  {
    v113 = *(_QWORD *)(a1 + 24);
    sub_1623A60(&v113, v24, 2);
    v25 = v19 + 48;
    if ( *(_QWORD *)(v19 + 48) )
    {
      sub_161E7C0(v19 + 48);
      v25 = v19 + 48;
    }
    v26 = v113;
    *(_QWORD *)(v19 + 48) = v113;
    if ( v26 )
      sub_1623210(&v113, v26, v25);
  }
  v27 = *(_DWORD *)(a1 + 136);
  v28 = a1 + 112;
  if ( !v27 )
  {
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_104;
  }
  v29 = *(_QWORD *)(a1 + 120);
  v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v31 = (_QWORD *)(v29 + 56LL * v30);
  v32 = *v31;
  if ( a2 != *v31 )
  {
    v87 = 1;
    v88 = 0;
    while ( v32 != -8 )
    {
      if ( !v88 && v32 == -16 )
        v88 = v31;
      v30 = (v27 - 1) & (v87 + v30);
      v31 = (_QWORD *)(v29 + 56LL * v30);
      v32 = *v31;
      if ( a2 == *v31 )
        goto LABEL_21;
      ++v87;
    }
    v89 = *(_DWORD *)(a1 + 128);
    if ( v88 )
      v31 = v88;
    ++*(_QWORD *)(a1 + 112);
    v90 = v89 + 1;
    if ( 4 * v90 < 3 * v27 )
    {
      if ( v27 - *(_DWORD *)(a1 + 132) - v90 > v27 >> 3 )
        goto LABEL_98;
      v110 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      sub_140F200(v28, v27);
      v98 = *(_DWORD *)(a1 + 136);
      if ( v98 )
      {
        v99 = v98 - 1;
        v100 = *(_QWORD *)(a1 + 120);
        v101 = 1;
        v102 = 0;
        LODWORD(v103) = v99 & v110;
        v31 = (_QWORD *)(v100 + 56LL * (v99 & v110));
        v104 = *v31;
        v90 = *(_DWORD *)(a1 + 128) + 1;
        if ( a2 != *v31 )
        {
          while ( v104 != -8 )
          {
            if ( v104 == -16 && !v102 )
              v102 = v31;
            v103 = v99 & (unsigned int)(v103 + v101);
            v31 = (_QWORD *)(v100 + 56 * v103);
            v104 = *v31;
            if ( a2 == *v31 )
              goto LABEL_98;
            ++v101;
          }
          if ( v102 )
            v31 = v102;
        }
        goto LABEL_98;
      }
      goto LABEL_135;
    }
LABEL_104:
    sub_140F200(v28, 2 * v27);
    v91 = *(_DWORD *)(a1 + 136);
    if ( v91 )
    {
      v92 = v91 - 1;
      v93 = *(_QWORD *)(a1 + 120);
      v94 = v92 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = (_QWORD *)(v93 + 56LL * v94);
      v95 = *v31;
      v90 = *(_DWORD *)(a1 + 128) + 1;
      if ( a2 != *v31 )
      {
        v96 = 1;
        v97 = 0;
        while ( v95 != -8 )
        {
          if ( !v97 && v95 == -16 )
            v97 = v31;
          v94 = v92 & (v94 + v96);
          v31 = (_QWORD *)(v93 + 56LL * v94);
          v95 = *v31;
          if ( a2 == *v31 )
            goto LABEL_98;
          ++v96;
        }
        if ( v97 )
          v31 = v97;
      }
LABEL_98:
      *(_DWORD *)(a1 + 128) = v90;
      if ( *v31 != -8 )
        --*(_DWORD *)(a1 + 132);
      *v31 = a2;
      v34 = v31 + 1;
      v31[1] = 6;
      v31[2] = 0;
      v31[3] = 0;
      v31[4] = 6;
      v31[5] = 0;
      v31[6] = 0;
      if ( !v8 )
      {
        v36 = v31 + 4;
        if ( !v19 )
          goto LABEL_35;
        goto LABEL_32;
      }
      goto LABEL_25;
    }
LABEL_135:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_21:
  v33 = v31[3];
  v34 = v31 + 1;
  if ( v8 != v33 )
  {
    if ( v33 != 0 && v33 != -8 && v33 != -16 )
    {
      sub_1649B30(v34);
      v34 = v31 + 1;
    }
LABEL_25:
    v31[3] = v8;
    if ( v8 != 0 && v8 != -8 && v8 != -16 )
      sub_164C220(v34);
  }
  v35 = v31[6];
  v36 = v31 + 4;
  if ( v19 == v35 )
    goto LABEL_35;
  if ( v35 != 0 && v35 != -8 && v35 != -16 )
  {
    sub_1649B30(v36);
    v36 = v31 + 4;
  }
LABEL_32:
  v31[6] = v19;
  if ( v19 != -8 && v19 != 0 && v19 != -16 )
    sub_164C220(v36);
LABEL_35:
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v37 = 0;
    v109 = a1 + 24;
    v108 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v38 = *(_QWORD *)(a2 - 8);
      else
        v38 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v39 = sub_157EE30(*(_QWORD *)(v37 + v38 + 24LL * *(unsigned int *)(a2 + 56) + 8));
      if ( !v39 )
        BUG();
      v40 = *(_QWORD *)(v39 + 16);
      *(_QWORD *)(a1 + 40) = v39;
      *(_QWORD *)(a1 + 32) = v40;
      v41 = *(_QWORD *)(v39 + 24);
      v116[0] = v41;
      if ( v41 )
      {
        sub_1623A60(v116, v41, 2);
        if ( !*(_QWORD *)(a1 + 24) )
          goto LABEL_43;
      }
      else if ( !*(_QWORD *)(a1 + 24) )
      {
        goto LABEL_45;
      }
      sub_161E7C0(v109);
LABEL_43:
      v42 = v116[0];
      *(_QWORD *)(a1 + 24) = v116[0];
      if ( v42 )
        sub_1623210(v116, v42, v109);
LABEL_45:
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v43 = *(_QWORD *)(a2 - 8);
      else
        v43 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v44 = sub_1410110(a1, *(_QWORD *)(v43 + 3 * v37));
      v46 = v44;
      if ( !v45 || !v44 )
      {
        v81 = sub_1599EF0(*(_QWORD *)(a1 + 96));
        sub_164D160(v112, v81);
        sub_15F20C0(v112, v81, v82, v83);
        v84 = sub_1599EF0(*(_QWORD *)(a1 + 96));
        sub_164D160(v111, v84);
        sub_15F20C0(v111, v84, v85, v86);
        return 0;
      }
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v47 = *(_QWORD *)(a2 - 8);
      else
        v47 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v48 = *(_QWORD *)(v37 + v47 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v49 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      if ( v49 == *(_DWORD *)(v8 + 56) )
      {
        v105 = v45;
        v107 = *(_QWORD *)(v37 + v47 + 24LL * *(unsigned int *)(a2 + 56) + 8);
        sub_15F55D0(v8);
        v45 = v105;
        v48 = v107;
        v49 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      }
      v50 = (v49 + 1) & 0xFFFFFFF;
      v51 = v50 | *(_DWORD *)(v8 + 20) & 0xF0000000;
      *(_DWORD *)(v8 + 20) = v51;
      if ( (v51 & 0x40000000) != 0 )
        v52 = *(_QWORD *)(v8 - 8);
      else
        v52 = v111 - 24 * v50;
      v53 = (__int64 *)(v52 + 24LL * (unsigned int)(v50 - 1));
      if ( *v53 )
      {
        v54 = v53[1];
        v55 = v53[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v55 = v54;
        if ( v54 )
          *(_QWORD *)(v54 + 16) = *(_QWORD *)(v54 + 16) & 3LL | v55;
      }
      *v53 = v46;
      v56 = *(_QWORD *)(v46 + 8);
      v53[1] = v56;
      if ( v56 )
        *(_QWORD *)(v56 + 16) = (unsigned __int64)(v53 + 1) | *(_QWORD *)(v56 + 16) & 3LL;
      v53[2] = v53[2] & 3 | (v46 + 8);
      *(_QWORD *)(v46 + 8) = v53;
      v57 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
        v58 = *(_QWORD *)(v8 - 8);
      else
        v58 = v111 - 24 * v57;
      *(_QWORD *)(v58 + 8LL * (unsigned int)(v57 - 1) + 24LL * *(unsigned int *)(v8 + 56) + 8) = v48;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v59 = *(_QWORD *)(a2 - 8);
      else
        v59 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v60 = *(_QWORD *)(v37 + v59 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v61 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
      if ( v61 == *(_DWORD *)(v19 + 56) )
      {
        v106 = v45;
        sub_15F55D0(v19);
        v45 = v106;
        v61 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
      }
      v62 = (v61 + 1) & 0xFFFFFFF;
      v63 = v62 | *(_DWORD *)(v19 + 20) & 0xF0000000;
      *(_DWORD *)(v19 + 20) = v63;
      if ( (v63 & 0x40000000) != 0 )
        v64 = *(_QWORD *)(v19 - 8);
      else
        v64 = v112 - 24 * v62;
      v65 = (_QWORD *)(v64 + 24LL * (unsigned int)(v62 - 1));
      if ( *v65 )
      {
        v66 = v65[1];
        v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v67 = v66;
        if ( v66 )
          *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
      }
      *v65 = v45;
      v68 = *(_QWORD *)(v45 + 8);
      v65[1] = v68;
      if ( v68 )
        *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
      v65[2] = v65[2] & 3LL | (v45 + 8);
      *(_QWORD *)(v45 + 8) = v65;
      v69 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
        v70 = *(_QWORD *)(v19 - 8);
      else
        v70 = v112 - 24 * v69;
      v37 += 8;
      *(_QWORD *)(v70 + 8LL * (unsigned int)(v69 - 1) + 24LL * *(unsigned int *)(v19 + 56) + 8) = v60;
    }
    while ( v108 != v37 );
  }
  v71 = sub_15F5600(v8);
  v72 = v71;
  if ( v71 )
  {
    v73 = v71;
    sub_164D160(v111, v71);
    v8 = v72;
    sub_15F20C0(v111, v73, v74, v75);
  }
  v76 = sub_15F5600(v19);
  if ( v76 )
  {
    v77 = v76;
    sub_164D160(v112, v76);
    sub_15F20C0(v112, v77, v78, v79);
  }
  return v8;
}
