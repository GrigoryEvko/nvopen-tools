// Function: sub_1DFDF40
// Address: 0x1dfdf40
//
void __fastcall sub_1DFDF40(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  _QWORD *v3; // r11
  __int64 v4; // r10
  _DWORD *v5; // rax
  int v6; // r9d
  __int64 v7; // rax
  unsigned int v8; // r12d
  unsigned __int64 v9; // r13
  int v11; // r14d
  int v12; // r15d
  __int64 v13; // r10
  unsigned __int64 v14; // r11
  __int64 v15; // rbx
  __int64 v16; // r14
  int v17; // r11d
  int v18; // r10d
  int v19; // r9d
  __int64 v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // r13
  int v23; // r8d
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r12
  __int64 v28; // r13
  int v29; // r15d
  int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // rdi
  int v33; // edx
  int v34; // eax
  __int64 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // edx
  int *v38; // rax
  __int64 v39; // rax
  int v41; // ebx
  int v42; // esi
  int *v43; // r14
  unsigned int v44; // edx
  int *v45; // rdi
  int v46; // ecx
  unsigned int v47; // ebx
  unsigned int v48; // edi
  int v49; // ebx
  __int64 v50; // rdx
  unsigned __int64 v51; // rsi
  unsigned __int64 v52; // rsi
  int *v55; // r9
  int v56; // eax
  __int64 v57; // rbx
  size_t v58; // r12
  __int64 v59; // rax
  int v60; // r9d
  __int64 v61; // r10
  void *v62; // r11
  _QWORD *v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rsi
  __int64 i; // rax
  int *v67; // r9
  int v68; // edx
  __int64 v69; // rax
  unsigned __int64 v70; // [rsp+18h] [rbp-C8h]
  __int64 v71; // [rsp+18h] [rbp-C8h]
  __int64 v72; // [rsp+18h] [rbp-C8h]
  _DWORD *v73; // [rsp+20h] [rbp-C0h]
  int v74; // [rsp+28h] [rbp-B8h]
  int v75; // [rsp+28h] [rbp-B8h]
  _QWORD *v76; // [rsp+28h] [rbp-B8h]
  int v77; // [rsp+28h] [rbp-B8h]
  _QWORD *v78; // [rsp+28h] [rbp-B8h]
  int v79; // [rsp+28h] [rbp-B8h]
  int v80; // [rsp+28h] [rbp-B8h]
  int v81; // [rsp+28h] [rbp-B8h]
  int v82; // [rsp+30h] [rbp-B0h]
  __int64 v83; // [rsp+30h] [rbp-B0h]
  int v84; // [rsp+30h] [rbp-B0h]
  int v85; // [rsp+30h] [rbp-B0h]
  int v86; // [rsp+30h] [rbp-B0h]
  int v87; // [rsp+3Ch] [rbp-A4h]
  int v88; // [rsp+3Ch] [rbp-A4h]
  int v89; // [rsp+3Ch] [rbp-A4h]
  int v90; // [rsp+3Ch] [rbp-A4h]
  int v91; // [rsp+3Ch] [rbp-A4h]
  int v92; // [rsp+3Ch] [rbp-A4h]
  unsigned __int64 v93; // [rsp+40h] [rbp-A0h]
  int v94; // [rsp+40h] [rbp-A0h]
  __int64 v95; // [rsp+40h] [rbp-A0h]
  __int64 v96; // [rsp+40h] [rbp-A0h]
  int v97; // [rsp+48h] [rbp-98h]
  int v98; // [rsp+48h] [rbp-98h]
  int v99; // [rsp+48h] [rbp-98h]
  __int64 v100; // [rsp+50h] [rbp-90h]
  __int64 v101; // [rsp+50h] [rbp-90h]
  unsigned __int64 v102; // [rsp+50h] [rbp-90h]
  int v103; // [rsp+50h] [rbp-90h]
  int v104; // [rsp+50h] [rbp-90h]
  void *v105; // [rsp+50h] [rbp-90h]
  __int64 v106; // [rsp+58h] [rbp-88h] BYREF
  int v107; // [rsp+64h] [rbp-7Ch] BYREF
  int *v108; // [rsp+68h] [rbp-78h] BYREF
  __int64 v109; // [rsp+70h] [rbp-70h] BYREF
  __int64 v110; // [rsp+78h] [rbp-68h]
  __int64 v111; // [rsp+80h] [rbp-60h]
  unsigned int v112; // [rsp+88h] [rbp-58h]
  __int64 v113; // [rsp+90h] [rbp-50h] BYREF
  __int64 v114; // [rsp+98h] [rbp-48h]
  __int64 v115; // [rsp+A0h] [rbp-40h]
  __int64 v116; // [rsp+A8h] [rbp-38h]

  v106 = a2;
  v2 = sub_1DFD350(a1 + 112, &v106);
  v3 = 0;
  v4 = a1;
  v5 = (_DWORD *)v2[1];
  v6 = v5[10];
  v73 = v5;
  v87 = v5[2];
  v82 = v5[3];
  if ( v6 )
  {
    v57 = (unsigned int)(v6 + 63) >> 6;
    v104 = v5[10];
    v58 = 8 * v57;
    v59 = malloc(8 * v57);
    v60 = v104;
    v61 = a1;
    v62 = (void *)v59;
    if ( !v59 )
    {
      if ( v58 || (v69 = malloc(1u), v62 = 0, v60 = v104, v61 = a1, !v69) )
      {
        v96 = v61;
        v99 = v60;
        v105 = v62;
        sub_16BD1C0("Allocation failed", 1u);
        v62 = v105;
        v60 = v99;
        v61 = v96;
      }
      else
      {
        v62 = (void *)v69;
      }
    }
    v95 = v61;
    v98 = v60;
    v63 = memcpy(v62, *((const void **)v73 + 3), v58);
    v6 = v98;
    v4 = v95;
    v3 = v63;
    v64 = (unsigned int)(v73[16] + 63) >> 6;
    if ( (unsigned int)v64 > (unsigned int)v57 )
      v64 = v57;
    if ( (_DWORD)v64 )
    {
      v65 = *((_QWORD *)v73 + 6);
      for ( i = 0; i != v64; ++i )
        v3[i] &= ~*(_QWORD *)(v65 + 8 * i);
    }
  }
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  if ( v6 )
  {
    v7 = 0;
    v8 = (unsigned int)(v6 - 1) >> 6;
    v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    while ( 1 )
    {
      _RDX = v3[v7];
      if ( v8 == (_DWORD)v7 )
        _RDX = v9 & v3[v7];
      if ( _RDX )
        break;
      if ( v8 + 1 == ++v7 )
        goto LABEL_8;
    }
    __asm { tzcnt   rdx, rdx }
    v41 = _RDX + ((_DWORD)v7 << 6);
    if ( v41 != -1 )
    {
      v42 = 0;
      v103 = v6;
      v43 = (int *)(*(_QWORD *)(v4 + 88) + 4LL * v41);
LABEL_69:
      ++v113;
LABEL_70:
      v71 = v4;
      v76 = v3;
      sub_136B240((__int64)&v113, 2 * v42);
      sub_1DF91F0((__int64)&v113, v43, &v108);
      v55 = v108;
      v3 = v76;
      v4 = v71;
      v56 = v115 + 1;
LABEL_78:
      LODWORD(v115) = v56;
      if ( *v55 != -1 )
        --HIDWORD(v115);
      *v55 = *v43;
LABEL_56:
      while ( 1 )
      {
        v47 = v41 + 1;
        if ( v103 == v47 )
          break;
        v48 = v47 >> 6;
        if ( v8 < v47 >> 6 )
          break;
        v49 = v47 & 0x3F;
        v50 = v48;
        v51 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49);
        if ( v49 == 0 )
          v51 = 0;
        v52 = ~v51;
        while ( 1 )
        {
          _RAX = v3[v50];
          if ( v48 == (_DWORD)v50 )
            _RAX = v52 & v3[v50];
          if ( v8 == (_DWORD)v50 )
            _RAX &= v9;
          if ( _RAX )
            break;
          if ( v8 < (unsigned int)++v50 )
            goto LABEL_8;
        }
        __asm { tzcnt   rax, rax }
        v41 = _RAX + ((_DWORD)v50 << 6);
        if ( v41 == -1 )
          break;
        v42 = v116;
        v43 = (int *)(*(_QWORD *)(v4 + 88) + 4LL * v41);
        if ( !(_DWORD)v116 )
          goto LABEL_69;
        v44 = (v116 - 1) & (37 * *v43);
        v45 = (int *)(v114 + 4LL * v44);
        v46 = *v45;
        if ( *v45 != *v43 )
        {
          v77 = 1;
          v55 = 0;
          while ( v46 != -1 )
          {
            if ( v46 != -2 || v55 )
              v45 = v55;
            v44 = (v116 - 1) & (v77 + v44);
            v46 = *(_DWORD *)(v114 + 4LL * v44);
            if ( *v43 == v46 )
              goto LABEL_56;
            ++v77;
            v55 = v45;
            v45 = (int *)(v114 + 4LL * v44);
          }
          if ( !v55 )
            v55 = v45;
          ++v113;
          v56 = v115 + 1;
          if ( 4 * ((int)v115 + 1) < (unsigned int)(3 * v116) )
          {
            if ( (int)v116 - (v56 + HIDWORD(v115)) <= (unsigned int)v116 >> 3 )
            {
              v72 = v4;
              v78 = v3;
              sub_136B240((__int64)&v113, v116);
              sub_1DF91F0((__int64)&v113, v43, &v108);
              v55 = v108;
              v4 = v72;
              v3 = v78;
              v56 = v115 + 1;
            }
            goto LABEL_78;
          }
          goto LABEL_70;
        }
      }
    }
  }
LABEL_8:
  v11 = 0;
  v12 = 0;
  v93 = (unsigned __int64)v3;
  v100 = v4;
  sub_1DF98C0(v4, v106, (__int64)&v109, (__int64)&v113);
  v13 = v100;
  v14 = v93;
  v15 = *(_QWORD *)(v106 + 32);
  if ( v106 + 24 == v15 )
    goto LABEL_37;
  v70 = v93;
  v16 = v100;
  v17 = v87;
  v18 = v82;
  v101 = v106 + 24;
  v19 = 0;
  do
  {
    v20 = *(_QWORD *)(v15 + 16);
    if ( !*(_WORD *)v20 || *(_WORD *)v20 == 45 )
      goto LABEL_34;
    v21 = *(unsigned __int8 *)(v20 + 4);
    v22 = *(int *)(v15 + 40);
    v97 = 0;
    v94 = 0;
    if ( v21 == (_DWORD)v22 )
      goto LABEL_20;
    v23 = v19;
    do
    {
      v24 = *(_QWORD *)(v15 + 32) + 40LL * v21;
      if ( *(_BYTE *)v24 )
        goto LABEL_18;
      if ( (*(_BYTE *)(v24 + 3) & 0x20) != 0 )
        goto LABEL_18;
      v25 = *(unsigned int *)(v24 + 8);
      if ( (int)v25 >= 0 || (*(_BYTE *)(v24 + 3) & 0x10) != 0 )
        goto LABEL_18;
      v107 = *(_DWORD *)(v24 + 8);
      if ( v112 )
      {
        v37 = (v112 - 1) & (37 * v25);
        v38 = (int *)(v110 + 16LL * v37);
        v89 = *v38;
        if ( *v38 == (_DWORD)v25 )
        {
LABEL_51:
          if ( *((_QWORD *)v38 + 1) == v15 )
          {
            v75 = v18;
            v84 = v23;
            v90 = v17;
            v39 = sub_21EA570(v25, *(_QWORD *)(v16 + 192), *(_QWORD *)(v16 + 176));
            v18 = v75;
            v94 += v39;
            v23 = v84;
            v17 = v90;
            v97 += HIDWORD(v39);
          }
          goto LABEL_18;
        }
        v79 = 1;
        v67 = 0;
        while ( v89 != -1 )
        {
          if ( v89 == -2 && !v67 )
            v67 = v38;
          v37 = (v112 - 1) & (v79 + v37);
          ++v79;
          v38 = (int *)(v110 + 16LL * v37);
          v89 = *v38;
          if ( (_DWORD)v25 == *v38 )
            goto LABEL_51;
        }
        if ( !v67 )
          v67 = v38;
        ++v109;
        v68 = v111 + 1;
        if ( 4 * ((int)v111 + 1) < 3 * v112 )
        {
          if ( v112 - HIDWORD(v111) - v68 <= v112 >> 3 )
          {
            v81 = v18;
            v86 = v23;
            v92 = v17;
            sub_1DF4FB0((__int64)&v109, v112);
            sub_1DF9340((__int64)&v109, &v107, &v108);
            v67 = v108;
            LODWORD(v25) = v107;
            v18 = v81;
            v23 = v86;
            v17 = v92;
            v68 = v111 + 1;
          }
          goto LABEL_99;
        }
      }
      else
      {
        ++v109;
      }
      v80 = v18;
      v85 = v23;
      v91 = v17;
      sub_1DF4FB0((__int64)&v109, 2 * v112);
      sub_1DF9340((__int64)&v109, &v107, &v108);
      v67 = v108;
      LODWORD(v25) = v107;
      v17 = v91;
      v23 = v85;
      v18 = v80;
      v68 = v111 + 1;
LABEL_99:
      LODWORD(v111) = v68;
      if ( *v67 != -1 )
        --HIDWORD(v111);
      *v67 = v25;
      *((_QWORD *)v67 + 1) = 0;
LABEL_18:
      ++v21;
    }
    while ( (_DWORD)v22 != v21 );
    v19 = v23;
    v22 = *(unsigned __int8 *)(*(_QWORD *)(v15 + 16) + 4LL);
LABEL_20:
    if ( (_DWORD)v22 )
    {
      v88 = v12;
      v26 = v16;
      v27 = 0;
      v28 = 40 * v22;
      v29 = v18;
      v30 = v17;
      do
      {
        v31 = v27 + *(_QWORD *)(v15 + 32);
        if ( !*(_BYTE *)v31 )
        {
          v32 = *(unsigned int *)(v31 + 8);
          if ( (int)v32 < 0 && **(_WORD **)(v15 + 16) != 9 && (*(_BYTE *)(v31 + 3) & 0x10) != 0 )
          {
            v74 = v19;
            v83 = v26;
            v36 = sub_21EA570(v32, *(_QWORD *)(v26 + 192), *(_QWORD *)(v26 + 176));
            v19 = v74;
            v26 = v83;
            v30 += v36;
            v29 += HIDWORD(v36);
          }
        }
        v27 += 40;
      }
      while ( v28 != v27 );
      v18 = v29;
      v12 = v88;
      v17 = v30;
      v16 = v26;
    }
    if ( v19 < v17 )
      v19 = v17;
    if ( v12 < v18 )
      v12 = v18;
    v17 -= v94;
    v18 -= v97;
LABEL_34:
    if ( (*(_BYTE *)v15 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v15 + 46) & 8) != 0 )
        v15 = *(_QWORD *)(v15 + 8);
    }
    v15 = *(_QWORD *)(v15 + 8);
  }
  while ( v101 != v15 );
  v14 = v70;
  v13 = v16;
  v11 = v12;
  v12 = v19;
LABEL_37:
  v33 = v12;
  v34 = v11;
  v102 = v14;
  if ( *v73 >= v12 )
    v33 = *v73;
  if ( v73[1] >= v11 )
    v34 = v73[1];
  *v73 = v33;
  v73[1] = v34;
  if ( *(_DWORD *)(v13 + 28) >= v34 )
    v34 = *(_DWORD *)(v13 + 28);
  if ( *(_DWORD *)(v13 + 24) >= v33 )
    v33 = *(_DWORD *)(v13 + 24);
  v35 = v114;
  *(_DWORD *)(v13 + 28) = v34;
  *(_DWORD *)(v13 + 24) = v33;
  j___libc_free_0(v35);
  j___libc_free_0(v110);
  _libc_free(v102);
}
