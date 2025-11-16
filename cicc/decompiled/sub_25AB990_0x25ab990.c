// Function: sub_25AB990
// Address: 0x25ab990
//
__int64 __fastcall sub_25AB990(__int64 a1)
{
  __int64 v1; // r12
  _BYTE *v2; // rax
  _BYTE *v3; // rax
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r8
  unsigned int v10; // esi
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r14
  __int16 v22; // r13
  __int64 v23; // rax
  unsigned __int8 v24; // r13
  unsigned __int16 v25; // ax
  __int16 v26; // ax
  __int64 v27; // rax
  unsigned __int8 v28; // dl
  unsigned __int8 *v29; // rax
  __int64 *v30; // r13
  __int64 *v31; // r15
  __int64 v32; // rsi
  __int64 v33; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rcx
  unsigned int v38; // edi
  _QWORD *v39; // rdx
  __int64 v40; // r8
  __int64 *v41; // rax
  __int64 v42; // rdx
  unsigned __int8 v43; // si
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned int v47; // ecx
  _QWORD *v48; // rdi
  __int64 v49; // r8
  unsigned int v50; // eax
  int v51; // eax
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  unsigned int v54; // ebx
  _QWORD *v55; // rax
  _QWORD *j; // rdx
  int v57; // edx
  int v58; // r11d
  _QWORD *v60; // rax
  int v61; // edi
  unsigned int v62; // edx
  __int64 v63; // r10
  int v64; // r8d
  _QWORD *v65; // rsi
  int v66; // r8d
  __int64 v67; // rdx
  __int64 v68; // r9
  _QWORD *v69; // r8
  __int64 v70; // [rsp+18h] [rbp-348h]
  __int64 v72; // [rsp+30h] [rbp-330h]
  __int64 i; // [rsp+38h] [rbp-328h]
  unsigned __int8 v74; // [rsp+47h] [rbp-319h]
  __int64 v75; // [rsp+48h] [rbp-318h]
  __int64 v76; // [rsp+58h] [rbp-308h]
  unsigned int v77; // [rsp+60h] [rbp-300h]
  __int64 v78; // [rsp+68h] [rbp-2F8h]
  int v79; // [rsp+68h] [rbp-2F8h]
  __int64 v80; // [rsp+68h] [rbp-2F8h]
  __int64 v81; // [rsp+68h] [rbp-2F8h]
  unsigned __int8 v82; // [rsp+7Fh] [rbp-2E1h] BYREF
  __int64 *v83; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v84; // [rsp+88h] [rbp-2D8h]
  _BYTE v85[16]; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v86; // [rsp+A0h] [rbp-2C0h] BYREF
  _QWORD *v87; // [rsp+A8h] [rbp-2B8h]
  __int64 v88; // [rsp+B0h] [rbp-2B0h]
  unsigned int v89; // [rsp+B8h] [rbp-2A8h]
  __int64 v90; // [rsp+C0h] [rbp-2A0h] BYREF
  char *v91; // [rsp+C8h] [rbp-298h]
  __int64 v92; // [rsp+D0h] [rbp-290h]
  int v93; // [rsp+D8h] [rbp-288h]
  char v94; // [rsp+DCh] [rbp-284h]
  char v95; // [rsp+E0h] [rbp-280h] BYREF
  _BYTE *v96; // [rsp+120h] [rbp-240h] BYREF
  __int64 v97; // [rsp+128h] [rbp-238h]
  _BYTE v98[560]; // [rsp+130h] [rbp-230h] BYREF

  v90 = 0;
  v91 = &v95;
  v92 = 8;
  v93 = 0;
  v94 = 1;
  v2 = sub_BA8CD0(a1, (__int64)"llvm.used", 9u, 0);
  sub_25AB6E0((__int64)v2, (unsigned __int8 *)&v90);
  v3 = sub_BA8CD0(a1, (__int64)"llvm.compiler.used", 0x12u, 0);
  sub_25AB6E0((__int64)v3, (unsigned __int8 *)&v90);
  v86 = 0;
  v96 = v98;
  v97 = 0x2000000000LL;
  v72 = a1 + 8;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  for ( i = 0; ; i = v75 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v75 = i;
    if ( v4 == v72 )
      break;
    do
    {
      while ( 1 )
      {
        v1 = v4;
        v4 = *(_QWORD *)(v4 + 8);
        v6 = v1 - 56;
        sub_AD0030(v1 - 56);
        if ( !*(_QWORD *)(v1 - 40) && (*(_BYTE *)(v1 - 24) & 0xFu) - 7 <= 1 )
          break;
        if ( sub_25AB620(v1 - 56, (__int64)&v90) )
          goto LABEL_7;
        v5 = *(_BYTE *)(v1 - 24) & 0xF;
        if ( ((v5 + 14) & 0xFu) <= 3 || ((v5 + 7) & 0xFu) <= 1 || (unsigned __int8)sub_25AB5A0(v1 - 56) )
          goto LABEL_7;
        v36 = v89;
        v37 = *(_QWORD *)(v1 - 88);
        if ( !v89 )
        {
          ++v86;
          goto LABEL_99;
        }
        v38 = (v89 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v39 = &v87[2 * v38];
        v40 = *v39;
        if ( v37 != *v39 )
        {
          v79 = 1;
          v60 = 0;
          while ( v40 != -4096 )
          {
            if ( v40 == -8192 && !v60 )
              v60 = v39;
            v38 = (v89 - 1) & (v79 + v38);
            v39 = &v87[2 * v38];
            v40 = *v39;
            if ( v37 == *v39 )
              goto LABEL_56;
            ++v79;
          }
          v36 = v89;
          if ( !v60 )
            v60 = v39;
          ++v86;
          v61 = v88 + 1;
          if ( 4 * ((int)v88 + 1) >= 3 * v89 )
          {
LABEL_99:
            v80 = v37;
            sub_25AB7B0((__int64)&v86, 2 * v36);
            if ( !v89 )
              goto LABEL_131;
            v37 = v80;
            v61 = v88 + 1;
            v62 = (v89 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
            v60 = &v87[2 * v62];
            v63 = *v60;
            if ( v80 != *v60 )
            {
              v64 = 1;
              v65 = 0;
              while ( v63 != -4096 )
              {
                if ( !v65 && v63 == -8192 )
                  v65 = v60;
                LODWORD(v1) = v64 + 1;
                v62 = (v89 - 1) & (v64 + v62);
                v60 = &v87[2 * v62];
                v63 = *v60;
                if ( v80 == *v60 )
                  goto LABEL_95;
                ++v64;
              }
LABEL_103:
              if ( v65 )
                v60 = v65;
            }
          }
          else if ( v89 - HIDWORD(v88) - v61 <= v89 >> 3 )
          {
            v77 = ((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4);
            v81 = v37;
            sub_25AB7B0((__int64)&v86, v89);
            if ( !v89 )
            {
LABEL_131:
              LODWORD(v88) = v88 + 1;
              BUG();
            }
            v66 = 1;
            v65 = 0;
            LODWORD(v67) = (v89 - 1) & v77;
            LODWORD(v1) = (_DWORD)v87;
            v60 = &v87[2 * (unsigned int)v67];
            v68 = *v60;
            v61 = v88 + 1;
            v37 = v81;
            if ( v81 != *v60 )
            {
              while ( v68 != -4096 )
              {
                if ( v68 == -8192 && !v65 )
                  v65 = v60;
                v67 = (v89 - 1) & ((_DWORD)v67 + v66);
                v60 = &v87[2 * v67];
                v68 = *v60;
                if ( v81 == *v60 )
                  goto LABEL_95;
                ++v66;
              }
              goto LABEL_103;
            }
          }
LABEL_95:
          LODWORD(v88) = v61;
          if ( *v60 != -4096 )
            --HIDWORD(v88);
          *v60 = v37;
          v41 = v60 + 1;
          *v41 = 0;
          goto LABEL_59;
        }
LABEL_56:
        v41 = v39 + 1;
        v42 = v39[1];
        if ( v42 )
        {
          v43 = *(_BYTE *)(v1 - 24);
          v44 = (*(_BYTE *)(v42 + 32) & 0xF) - 7;
          if ( ((v43 + 9) & 0xFu) <= 1 )
          {
            if ( v44 > 1 )
              goto LABEL_7;
LABEL_77:
            if ( v43 >> 6 != 2 )
              goto LABEL_7;
            goto LABEL_59;
          }
          if ( v44 > 1 )
            goto LABEL_77;
        }
LABEL_59:
        *v41 = v6;
LABEL_7:
        if ( v4 == v72 )
          goto LABEL_11;
      }
      sub_B30290(v1 - 56);
      ++v75;
    }
    while ( v4 != v72 );
LABEL_11:
    v7 = *(_QWORD *)(a1 + 16);
    while ( v7 != v72 )
    {
      v1 = v7;
      v7 = *(_QWORD *)(v7 + 8);
      v8 = v1 - 56;
      if ( !sub_25AB620(v1 - 56, (__int64)&v90) && (*(_BYTE *)(v1 - 24) & 0xFu) - 7 <= 1 )
      {
        v9 = *(_QWORD *)(v1 - 88);
        if ( v89 )
        {
          v10 = (v89 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v11 = &v87[2 * v10];
          v12 = *v11;
          if ( v9 == *v11 )
          {
LABEL_16:
            if ( v11 != &v87[2 * v89] )
            {
              v13 = v11[1];
              if ( v13 != v8 && (*(_BYTE *)(v1 - 24) >> 6 == 2 || *(_BYTE *)(v13 + 32) >> 6 == 2) )
              {
                v78 = v11[1];
                if ( !(unsigned __int8)sub_25AB5A0(v1 - 56) )
                {
                  v15 = v78;
                  if ( *(_BYTE *)(v1 - 24) >> 6 != 2 )
                    *(_BYTE *)(v78 + 32) &= 0x3Fu;
                  v16 = (unsigned int)v97;
                  v17 = (unsigned int)v97 + 1LL;
                  if ( v17 > HIDWORD(v97) )
                  {
                    sub_C8D5F0((__int64)&v96, v98, v17, 0x10u, v78, v14);
                    v16 = (unsigned int)v97;
                    v15 = v78;
                  }
                  v18 = (__int64 *)&v96[16 * v16];
                  *v18 = v8;
                  v18[1] = v15;
                  LODWORD(v97) = v97 + 1;
                }
              }
            }
          }
          else
          {
            v57 = 1;
            while ( v12 != -4096 )
            {
              v58 = v57 + 1;
              v10 = (v89 - 1) & (v57 + v10);
              v11 = &v87[2 * v10];
              v12 = *v11;
              if ( v9 == *v11 )
                goto LABEL_16;
              v57 = v58;
            }
          }
        }
      }
    }
    v19 = (unsigned int)v97;
    if ( !(_DWORD)v97 )
      goto LABEL_45;
LABEL_28:
    v70 = v19;
    v20 = 0;
    v76 = 16 * v19;
    while ( 2 )
    {
      v1 = *(_QWORD *)&v96[v20 + 8];
      v21 = *(_QWORD *)&v96[v20];
      v22 = (*(_WORD *)(v1 + 34) >> 1) & 0x3F;
      if ( (*(_BYTE *)(v21 + 34) & 0x7E) != 0 )
      {
        if ( !v22 )
        {
          v23 = sub_B2F730(*(_QWORD *)&v96[v20 + 8]);
          v24 = sub_AE5270(v23, v1);
          goto LABEL_32;
        }
LABEL_61:
        v24 = v22 - 1;
        v45 = sub_B2F730(*(_QWORD *)&v96[v20 + 8]);
        sub_AE5270(v45, v1);
LABEL_32:
        v25 = *(_WORD *)(v21 + 34);
        v82 = v24;
        v26 = (v25 >> 1) & 0x3F;
        if ( v26 )
        {
          v74 = v26 - 1;
          v27 = sub_B2F730(v21);
          sub_AE5270(v27, v21);
          v28 = v74;
        }
        else
        {
          v46 = sub_B2F730(v21);
          v28 = sub_AE5270(v46, v21);
        }
        v29 = (unsigned __int8 *)&v83;
        LOBYTE(v83) = v28;
        if ( v28 < v24 )
          v29 = &v82;
        sub_B2F770(v1, *v29);
      }
      else if ( v22 )
      {
        goto LABEL_61;
      }
      v83 = (__int64 *)v85;
      v84 = 0x100000000LL;
      sub_B92230(v21, (__int64)&v83);
      v30 = &v83[(unsigned int)v84];
      if ( v83 != v30 )
      {
        v31 = v83;
        do
        {
          v32 = *v31++;
          sub_B996C0(v1, v32);
        }
        while ( v30 != v31 );
        v30 = v83;
      }
      if ( v30 != (__int64 *)v85 )
        _libc_free((unsigned __int64)v30);
      v20 += 16;
      sub_BD84D0(v21, v1);
      sub_B30290(v21);
      if ( v76 != v20 )
        continue;
      break;
    }
    v75 += v70;
LABEL_45:
    if ( i == v75 )
      goto LABEL_84;
    ++v86;
    LODWORD(v97) = 0;
    if ( (_DWORD)v88 )
    {
      v47 = 4 * v88;
      v33 = v89;
      if ( (unsigned int)(4 * v88) < 0x40 )
        v47 = 64;
      if ( v89 <= v47 )
      {
LABEL_49:
        v34 = v87;
        v35 = &v87[2 * v33];
        if ( v87 != v35 )
        {
          do
          {
            *v34 = -4096;
            v34 += 2;
          }
          while ( v35 != v34 );
        }
        v88 = 0;
        continue;
      }
      v48 = v87;
      v49 = 2LL * v89;
      if ( (_DWORD)v88 == 1 )
      {
        v1 = 2048;
        v54 = 128;
      }
      else
      {
        _BitScanReverse(&v50, v88 - 1);
        v51 = 1 << (33 - (v50 ^ 0x1F));
        if ( v51 < 64 )
          v51 = 64;
        if ( v89 == v51 )
        {
          v88 = 0;
          v69 = &v87[v49];
          do
          {
            if ( v48 )
              *v48 = -4096;
            v48 += 2;
          }
          while ( v69 != v48 );
          continue;
        }
        v52 = (4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1);
        v53 = ((v52 | (v52 >> 2)) >> 4) | v52 | (v52 >> 2) | ((((v52 | (v52 >> 2)) >> 4) | v52 | (v52 >> 2)) >> 8);
        v54 = (v53 | (v53 >> 16)) + 1;
        v1 = 16 * ((v53 | (v53 >> 16)) + 1);
      }
      sub_C7D6A0((__int64)v87, v49 * 8, 8);
      v89 = v54;
      v55 = (_QWORD *)sub_C7D670(v1, 8);
      v88 = 0;
      v87 = v55;
      for ( j = &v55[2 * v89]; j != v55; v55 += 2 )
      {
        if ( v55 )
          *v55 = -4096;
      }
      continue;
    }
    if ( HIDWORD(v88) )
    {
      v33 = v89;
      if ( v89 <= 0x40 )
        goto LABEL_49;
      sub_C7D6A0((__int64)v87, 16LL * v89, 8);
      v87 = 0;
      v88 = 0;
      v89 = 0;
    }
  }
  v19 = (unsigned int)v97;
  if ( (_DWORD)v97 )
    goto LABEL_28;
LABEL_84:
  if ( v96 != v98 )
    _libc_free((unsigned __int64)v96);
  sub_C7D6A0((__int64)v87, 16LL * v89, 8);
  if ( !v94 )
    _libc_free((unsigned __int64)v91);
  LOBYTE(v1) = v75 != 0;
  return (unsigned int)v1;
}
