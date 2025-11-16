// Function: sub_384D760
// Address: 0x384d760
//
__int64 __fastcall sub_384D760(__int64 a1, _QWORD *a2)
{
  __int64 **v2; // rbx
  __int64 **v3; // rax
  char **v4; // r15
  __int64 v5; // r13
  unsigned __int64 *v6; // r14
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  char *v12; // r12
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  char *v15; // rax
  char *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r14
  unsigned __int8 v20; // al
  __int64 v21; // r12
  __int64 v22; // r12
  unsigned __int64 v23; // r13
  unsigned __int64 *v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned int v27; // edi
  __int64 *v28; // rcx
  __int64 v29; // r9
  __int64 *v30; // r13
  unsigned __int64 v31; // rsi
  __int64 v32; // rdx
  char v33; // di
  __int64 v34; // rcx
  bool v35; // di
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rsi
  int v39; // ecx
  int v40; // r11d
  int v41; // r8d
  unsigned __int8 v42; // dl
  unsigned __int64 v43; // rdi
  __int64 v44; // rcx
  unsigned __int64 v45; // rdi
  int v47; // ecx
  unsigned __int64 v48; // rsi
  __int64 v49; // rsi
  unsigned int v50; // ecx
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  __int64 *v53; // rsi
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 *v56; // rcx
  __int64 *v57; // rdx
  bool v58; // zf
  __int64 *v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // r12
  unsigned __int64 v62; // r13
  __int64 v63; // r9
  unsigned __int64 *v64; // rax
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rdi
  _QWORD *v67; // rdi
  unsigned int v68; // eax
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rax
  unsigned int v72; // ebx
  unsigned __int64 v73; // r12
  _QWORD *v74; // rax
  _QWORD *i; // rdx
  _QWORD *v76; // rax
  int v77; // r8d
  unsigned __int64 *v78; // rdi
  int v79; // ecx
  int v80; // r8d
  unsigned __int64 *v81; // rdi
  unsigned int v82; // esi
  unsigned __int64 v83; // r9
  unsigned int v84; // esi
  unsigned __int64 v85; // r9
  int v86; // r8d
  unsigned int v87; // [rsp+20h] [rbp-C0h]
  unsigned int v88; // [rsp+24h] [rbp-BCh]
  __int64 **v89; // [rsp+28h] [rbp-B8h]
  char v90; // [rsp+30h] [rbp-B0h]
  unsigned int v91; // [rsp+34h] [rbp-ACh]
  __int64 v92; // [rsp+38h] [rbp-A8h]
  unsigned int v93; // [rsp+38h] [rbp-A8h]
  __int64 **v94; // [rsp+40h] [rbp-A0h]
  unsigned int v95; // [rsp+48h] [rbp-98h]
  unsigned __int8 v96; // [rsp+4Fh] [rbp-91h]
  __int64 v97; // [rsp+50h] [rbp-90h]
  __int64 v98; // [rsp+50h] [rbp-90h]
  __int64 v100; // [rsp+68h] [rbp-78h] BYREF
  __int64 v101; // [rsp+70h] [rbp-70h] BYREF
  _QWORD *v102; // [rsp+78h] [rbp-68h]
  __int64 v103; // [rsp+80h] [rbp-60h]
  unsigned int v104; // [rsp+88h] [rbp-58h]
  unsigned __int64 v105[2]; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 v106; // [rsp+A0h] [rbp-40h]
  unsigned __int64 v107; // [rsp+A8h] [rbp-38h]

  v2 = *(__int64 ***)(a1 + 24);
  v3 = *(__int64 ***)(a1 + 16);
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v89 = v2;
  if ( v3 == v2 )
  {
    v96 = 0;
    v45 = 0;
    goto LABEL_89;
  }
  v94 = v3;
  v96 = 0;
  v90 = 0;
  do
  {
    v4 = (char **)*v94;
    v5 = **v94;
    if ( !v5 || sub_15E4F60(**v94) )
      goto LABEL_3;
    v6 = (unsigned __int64 *)v4[1];
    v88 = 0;
    v87 = 0;
    if ( v6 == (unsigned __int64 *)v4[2] )
      goto LABEL_27;
    v97 = v5;
    v7 = (unsigned __int64 *)v4[2];
    do
    {
      v8 = v6[2];
      if ( !v8 )
        goto LABEL_11;
      if ( v104 )
      {
        v9 = (v104 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v10 = v102[2 * v9];
        if ( v8 == v10 )
          goto LABEL_11;
        v41 = 1;
        while ( v10 != -8 )
        {
          v9 = (v104 - 1) & (v41 + v9);
          v10 = v102[2 * v9];
          if ( v8 == v10 )
            goto LABEL_11;
          ++v41;
        }
      }
      v42 = *(_BYTE *)(v8 + 16);
      if ( v42 > 0x17u )
      {
        if ( v42 == 78 )
        {
          v43 = v8 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_11;
          v44 = v8 | 4;
        }
        else
        {
          if ( v42 != 29 )
            goto LABEL_11;
          v43 = v8 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_11;
          v44 = v8 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v47 = (v44 >> 2) & 1;
        if ( v47 )
          v48 = v43 - 24;
        else
          v48 = v43 - 72;
        if ( *(_BYTE *)(*(_QWORD *)v48 + 16LL) )
          goto LABEL_98;
        if ( (_BYTE)v47 )
          v53 = (__int64 *)(v43 - 24);
        else
          v53 = (__int64 *)(v43 - 72);
        v49 = *v53;
        if ( *(_BYTE *)(v49 + 16) )
          BUG();
        if ( (*(_BYTE *)(v49 + 33) & 0x20) == 0 )
          goto LABEL_98;
        if ( (_BYTE)v47 )
          v66 = v43 - 24;
        else
          v66 = v43 - 72;
        if ( *(_BYTE *)(*(_QWORD *)v66 + 16LL) )
          BUG();
        if ( !sub_15E1830(*(_DWORD *)(*(_QWORD *)v66 + 36LL)) )
        {
          v8 = v6[2];
          v42 = *(_BYTE *)(v8 + 16);
          if ( v42 <= 0x17u )
            goto LABEL_100;
LABEL_98:
          if ( v42 == 78 )
          {
            v54 = v8 | 4;
          }
          else
          {
            if ( v42 != 29 )
            {
LABEL_100:
              v6 += 4;
              continue;
            }
            v54 = v8 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v55 = v54 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v54 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_100;
          v56 = (__int64 *)(v55 - 24);
          v57 = (__int64 *)(v55 - 72);
          v58 = (v54 & 4) == 0;
          v59 = v56;
          if ( v58 )
            v59 = v57;
          v60 = *v59;
          if ( !*(_BYTE *)(v60 + 16) && (*(_BYTE *)(v60 + 33) & 0x20) != 0 )
            goto LABEL_100;
          v105[0] = 6;
          v105[1] = 0;
          v106 = v6[2];
          v61 = v106;
          if ( v106 != -8 && v106 != 0 && v106 != -16 )
          {
            sub_1649AC0(v105, *v6 & 0xFFFFFFFFFFFFFFF8LL);
            v61 = v106;
          }
          v62 = v6[3];
          v107 = v62;
          if ( v104 )
          {
            v93 = ((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4);
            v63 = (v104 - 1) & v93;
            v64 = &v102[2 * v63];
            v65 = *v64;
            if ( v61 == *v64 )
            {
LABEL_122:
              if ( v65 != 0 && v65 != -8 && v65 != -16 )
                sub_1649B30(v105);
              goto LABEL_100;
            }
            v77 = 1;
            v78 = 0;
            while ( v65 != -8 )
            {
              if ( v65 == -16 && !v78 )
                v78 = v64;
              LODWORD(v63) = (v104 - 1) & (v77 + v63);
              v64 = &v102[2 * (unsigned int)v63];
              v65 = *v64;
              if ( v61 == *v64 )
                goto LABEL_122;
              ++v77;
            }
            if ( v78 )
              v64 = v78;
            ++v101;
            v79 = v103 + 1;
            if ( 4 * ((int)v103 + 1) < 3 * v104 )
            {
              if ( v104 - HIDWORD(v103) - v79 <= v104 >> 3 )
              {
                sub_384D5A0((__int64)&v101, v104);
                if ( !v104 )
                  goto LABEL_188;
                v80 = 1;
                v79 = v103 + 1;
                v81 = 0;
                v82 = (v104 - 1) & v93;
                v64 = &v102[2 * v82];
                v83 = *v64;
                if ( v61 != *v64 )
                {
                  while ( v83 != -8 )
                  {
                    if ( !v81 && v83 == -16 )
                      v81 = v64;
                    v82 = (v104 - 1) & (v80 + v82);
                    v64 = &v102[2 * v82];
                    v83 = *v64;
                    if ( v61 == *v64 )
                      goto LABEL_159;
                    ++v80;
                  }
                  goto LABEL_165;
                }
              }
              goto LABEL_159;
            }
          }
          else
          {
            ++v101;
          }
          sub_384D5A0((__int64)&v101, 2 * v104);
          if ( !v104 )
          {
LABEL_188:
            LODWORD(v103) = v103 + 1;
            BUG();
          }
          v84 = (v104 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
          v79 = v103 + 1;
          v64 = &v102[2 * v84];
          v85 = *v64;
          if ( v61 != *v64 )
          {
            v86 = 1;
            v81 = 0;
            while ( v85 != -8 )
            {
              if ( v85 == -16 && !v81 )
                v81 = v64;
              v84 = (v104 - 1) & (v86 + v84);
              v64 = &v102[2 * v84];
              v85 = *v64;
              if ( v61 == *v64 )
                goto LABEL_159;
              ++v86;
            }
LABEL_165:
            if ( v81 )
              v64 = v81;
          }
LABEL_159:
          LODWORD(v103) = v79;
          if ( *v64 != -8 )
            --HIDWORD(v103);
          *v64 = v61;
          v64[1] = v62;
          v65 = v106;
          goto LABEL_122;
        }
      }
LABEL_11:
      v11 = v6[3];
      if ( *(_QWORD *)v11 )
        ++v88;
      else
        ++v87;
      --*(_DWORD *)(v11 + 32);
      v12 = v4[2];
      v13 = v6[2];
      v14 = *((_QWORD *)v12 - 2);
      if ( v13 != v14 )
      {
        if ( v13 != 0 && v13 != -8 && v13 != -16 )
        {
          sub_1649B30(v6);
          v14 = *((_QWORD *)v12 - 2);
        }
        v6[2] = v14;
        if ( v14 != 0 && v14 != -8 && v14 != -16 )
          sub_1649AC0(v6, *((_QWORD *)v12 - 4) & 0xFFFFFFFFFFFFFFF8LL);
      }
      v6[3] = *((_QWORD *)v12 - 1);
      v15 = v4[2];
      v16 = v15 - 32;
      v4[2] = v15 - 32;
      v17 = *((_QWORD *)v15 - 2);
      if ( v17 != -8 && v17 != 0 && v17 != -16 )
        sub_1649B30(v16);
      if ( v6 + 4 == v7 )
        break;
      v7 = (unsigned __int64 *)v4[2];
    }
    while ( v7 != v6 );
    v5 = v97;
LABEL_27:
    v92 = v5 + 72;
    v98 = *(_QWORD *)(v5 + 80);
    if ( v98 != v5 + 72 )
    {
      v91 = 0;
      v95 = 0;
      while ( 1 )
      {
        if ( !v98 )
          BUG();
        v18 = *(_QWORD *)(v98 + 24);
        v19 = v98 + 16;
        if ( v18 != v98 + 16 )
          break;
LABEL_52:
        v98 = *(_QWORD *)(v98 + 8);
        if ( v92 == v98 )
        {
          v35 = v96;
          if ( v88 < v95 && v87 > v91 )
            v35 = v88 < v95 && v87 > v91;
          v96 = v35;
          goto LABEL_56;
        }
      }
      while ( 1 )
      {
LABEL_34:
        if ( !v18 )
          BUG();
        v20 = *(_BYTE *)(v18 - 8);
        v21 = v18 - 24;
        if ( v20 <= 0x17u )
          goto LABEL_33;
        if ( v20 == 78 )
        {
          v22 = v21 | 4;
        }
        else
        {
          if ( v20 != 29 )
            goto LABEL_33;
          v22 = v21 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_33;
        v24 = (unsigned __int64 *)(v23 - 24);
        if ( (v22 & 4) == 0 )
          v24 = (unsigned __int64 *)(v23 - 72);
        v25 = *v24;
        if ( *(_BYTE *)(*v24 + 16) )
        {
          v26 = v104;
          v25 = 0;
          if ( !v104 )
            goto LABEL_62;
        }
        else
        {
          if ( (*(_BYTE *)(v25 + 33) & 0x20) != 0 )
            goto LABEL_33;
          v26 = v104;
          if ( !v104 )
            goto LABEL_62;
        }
        v27 = (v26 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v28 = &v102[2 * v27];
        v29 = *v28;
        if ( v23 != *v28 )
          break;
LABEL_45:
        if ( v28 == &v102[2 * v26] )
          goto LABEL_62;
        *v28 = -16;
        v30 = (__int64 *)v28[1];
        LODWORD(v103) = v103 - 1;
        ++HIDWORD(v103);
        v31 = *v24;
        v32 = *v30;
        if ( *(_BYTE *)(*v24 + 16) )
        {
          if ( !v32 )
            goto LABEL_33;
          v34 = a2[8];
        }
        else
        {
          if ( v32 == v31 )
            goto LABEL_33;
          v33 = v96;
          v34 = sub_1399010(a2, v31);
          if ( !*v30 )
            v33 = 1;
          v96 = v33;
        }
        sub_13986C0((__int64)v4, v22, v22, v34);
        v18 = *(_QWORD *)(v18 + 8);
        if ( v19 == v18 )
          goto LABEL_52;
      }
      v39 = 1;
      while ( v29 != -8 )
      {
        v40 = v39 + 1;
        v27 = (v26 - 1) & (v39 + v27);
        v28 = &v102[2 * v27];
        v29 = *v28;
        if ( v23 == *v28 )
          goto LABEL_45;
        v39 = v40;
      }
LABEL_62:
      if ( v25 )
      {
        v37 = sub_1399010(a2, v25);
        ++v95;
      }
      else
      {
        ++v91;
        v37 = a2[8];
      }
      v105[0] = v22 & 0xFFFFFFFFFFFFFFF8LL;
      v38 = (__int64)v4[2];
      v100 = v37;
      if ( (char *)v38 == v4[3] )
      {
        sub_1398B10(v4 + 1, (char *)v38, v105, &v100);
        v37 = v100;
      }
      else
      {
        if ( v38 )
        {
          *(_QWORD *)v38 = 6;
          *(_QWORD *)(v38 + 8) = 0;
          *(_QWORD *)(v38 + 16) = v23;
          if ( (v22 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
            sub_164C220(v38);
          v37 = v100;
          *(_QWORD *)(v38 + 24) = v100;
          v38 = (__int64)v4[2];
        }
        v4[2] = (char *)(v38 + 32);
      }
      ++*(_DWORD *)(v37 + 32);
LABEL_33:
      v18 = *(_QWORD *)(v18 + 8);
      if ( v19 == v18 )
        goto LABEL_52;
      goto LABEL_34;
    }
LABEL_56:
    if ( (v90 & 0xF) != 0xF )
      goto LABEL_3;
    ++v101;
    if ( !(_DWORD)v103 )
    {
      if ( !HIDWORD(v103) )
        goto LABEL_3;
      v36 = v104;
      if ( v104 > 0x40 )
      {
        j___libc_free_0((unsigned __int64)v102);
        v102 = 0;
        v103 = 0;
        v104 = 0;
        goto LABEL_3;
      }
LABEL_104:
      v51 = v102;
      v52 = &v102[2 * v36];
      if ( v102 != v52 )
      {
        do
        {
          *v51 = -8;
          v51 += 2;
        }
        while ( v52 != v51 );
      }
      v103 = 0;
      goto LABEL_3;
    }
    v50 = 4 * v103;
    v36 = v104;
    if ( (unsigned int)(4 * v103) < 0x40 )
      v50 = 64;
    if ( v50 >= v104 )
      goto LABEL_104;
    v67 = v102;
    if ( (_DWORD)v103 == 1 )
    {
      v73 = 2048;
      v72 = 128;
LABEL_141:
      j___libc_free_0((unsigned __int64)v102);
      v104 = v72;
      v74 = (_QWORD *)sub_22077B0(v73);
      v103 = 0;
      v102 = v74;
      for ( i = &v74[2 * v104]; i != v74; v74 += 2 )
      {
        if ( v74 )
          *v74 = -8;
      }
      goto LABEL_3;
    }
    _BitScanReverse(&v68, v103 - 1);
    v69 = (unsigned int)(1 << (33 - (v68 ^ 0x1F)));
    if ( (int)v69 < 64 )
      v69 = 64;
    if ( (_DWORD)v69 != v104 )
    {
      v70 = (4 * (int)v69 / 3u + 1) | ((unsigned __int64)(4 * (int)v69 / 3u + 1) >> 1);
      v71 = ((v70 | (v70 >> 2)) >> 4) | v70 | (v70 >> 2) | ((((v70 | (v70 >> 2)) >> 4) | v70 | (v70 >> 2)) >> 8);
      v72 = (v71 | (v71 >> 16)) + 1;
      v73 = 16 * ((v71 | (v71 >> 16)) + 1);
      goto LABEL_141;
    }
    v103 = 0;
    v76 = &v102[2 * v69];
    do
    {
      if ( v67 )
        *v67 = -8;
      v67 += 2;
    }
    while ( v76 != v67 );
LABEL_3:
    ++v94;
    ++v90;
  }
  while ( v89 != v94 );
  v45 = (unsigned __int64)v102;
LABEL_89:
  j___libc_free_0(v45);
  return v96;
}
