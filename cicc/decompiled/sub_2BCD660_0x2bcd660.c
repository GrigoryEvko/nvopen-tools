// Function: sub_2BCD660
// Address: 0x2bcd660
//
__int64 __fastcall sub_2BCD660(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v6; // rbx
  __int64 v7; // r9
  int v8; // edx
  unsigned __int64 v9; // r13
  __int64 v10; // r12
  _BYTE *v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE *v17; // rdi
  unsigned __int8 **v18; // r11
  unsigned __int8 **v19; // r15
  unsigned int v20; // eax
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // rbx
  int v23; // edx
  __int64 v24; // rcx
  unsigned __int8 **i; // r12
  int v26; // edi
  unsigned int v27; // eax
  unsigned __int8 *v28; // rsi
  unsigned __int8 **v29; // r13
  unsigned __int8 *v30; // rdi
  unsigned __int8 *v31; // rsi
  unsigned __int8 *v32; // rax
  unsigned int v33; // ebx
  unsigned __int8 v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned int v39; // eax
  unsigned int v40; // eax
  __int64 v41; // rax
  char v42; // al
  char v43; // cl
  __int64 *v44; // rbx
  int v45; // ecx
  unsigned int v46; // edx
  __int64 v47; // rdi
  __int64 v48; // r12
  int v49; // edx
  __int64 v50; // rsi
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  int v53; // r10d
  __int64 *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // r10d
  int v58; // edx
  __int64 *v59; // rbx
  __int64 *v60; // r10
  int v61; // ecx
  unsigned int v62; // edx
  __int64 v63; // rdi
  __int64 v64; // r12
  int v65; // ecx
  __int64 v66; // rsi
  __int64 v67; // rdi
  __int64 v68; // rsi
  unsigned __int8 v70; // [rsp+Fh] [rbp-181h]
  __int64 v71; // [rsp+10h] [rbp-180h]
  unsigned __int8 v72; // [rsp+10h] [rbp-180h]
  __int64 v73; // [rsp+30h] [rbp-160h]
  unsigned __int8 **v75; // [rsp+40h] [rbp-150h]
  unsigned __int8 **v76; // [rsp+40h] [rbp-150h]
  unsigned __int8 **v77; // [rsp+40h] [rbp-150h]
  unsigned __int8 v78; // [rsp+40h] [rbp-150h]
  unsigned __int8 **v79; // [rsp+40h] [rbp-150h]
  unsigned __int8 **v80; // [rsp+40h] [rbp-150h]
  unsigned __int8 **v81; // [rsp+40h] [rbp-150h]
  __int64 *v82; // [rsp+40h] [rbp-150h]
  unsigned __int8 v83; // [rsp+50h] [rbp-140h]
  unsigned __int8 **v84; // [rsp+50h] [rbp-140h]
  __int64 v85; // [rsp+58h] [rbp-138h]
  __int64 v86; // [rsp+68h] [rbp-128h] BYREF
  __int64 v87[2]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v88; // [rsp+80h] [rbp-110h] BYREF
  __int64 v89; // [rsp+88h] [rbp-108h]
  __int64 v90; // [rsp+90h] [rbp-100h]
  __int64 v91; // [rsp+98h] [rbp-F8h]
  _BYTE *v92; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v93; // [rsp+A8h] [rbp-E8h]
  _BYTE v94[48]; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 *v95; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v96; // [rsp+E8h] [rbp-A8h]
  _BYTE v97[48]; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 *v98; // [rsp+120h] [rbp-70h] BYREF
  __int64 v99; // [rsp+128h] [rbp-68h]
  _BYTE v100[96]; // [rsp+130h] [rbp-60h] BYREF

  v3 = *(unsigned int *)(a1 + 112);
  v4 = *(_QWORD *)(a1 + 104);
  v86 = a1;
  v88 = 0;
  v89 = 0;
  v85 = v4;
  v90 = 0;
  v91 = 0;
  v73 = v4 + 88 * v3;
  if ( v4 == v73 )
  {
    v70 = 0;
    v67 = 0;
    v68 = 0;
  }
  else
  {
    v70 = 0;
    do
    {
      if ( *(_DWORD *)(v85 + 16) > 1u )
      {
        v6 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v85 + 8) - 64LL) + 8LL);
        if ( (_BYTE)qword_5010508 && *(_BYTE *)(v6 + 8) == 17 )
          v6 = **(_QWORD **)(v6 + 16);
        if ( (unsigned __int8)sub_BCBCB0(v6) && (*(_BYTE *)(v6 + 8) & 0xFD) != 4 )
        {
          v8 = 0;
          v9 = *(unsigned int *)(v85 + 16);
          v10 = *(_QWORD *)(v85 + 8);
          v11 = v94;
          v93 = 0x600000000LL;
          v92 = v94;
          v12 = 8 * v9;
          v13 = v9;
          if ( v9 > 6 )
          {
            sub_C8D5F0((__int64)&v92, v94, v9, 8u, v12, v7);
            v8 = v93;
            v12 = 8 * v9;
            v11 = &v92[8 * (unsigned int)v93];
          }
          if ( v12 )
          {
            v14 = v12 - 8 * v9 + v10;
            do
            {
              v11 += 8;
              *((_QWORD *)v11 - 1) = *(_QWORD *)(v14 + 8 * v13-- - 8);
            }
            while ( v13 );
            v8 = v93;
          }
          LODWORD(v93) = v8 + v9;
          sub_2BC67B0((__int64)&v92, (__int64 (__fastcall *)(__int64, __int64, __int64))sub_2B08970, (__int64)&v86);
          v95 = (__int64 *)v97;
          v17 = v92;
          v96 = 0x600000000LL;
          v99 = 0x600000000LL;
          v98 = (__int64 *)v100;
          if ( v92 != &v92[8 * (unsigned int)v93] )
          {
            v18 = (unsigned __int8 **)&v92[8 * (unsigned int)v93];
            v83 = 0;
            v19 = (unsigned __int8 **)v92;
            while ( 1 )
            {
              v22 = *v19;
              if ( !*v19 )
                goto LABEL_19;
              v23 = *(_DWORD *)(a2 + 2000);
              v24 = *(_QWORD *)(a2 + 1984);
              if ( v23 )
              {
                v15 = 1;
                v20 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                v21 = *(unsigned __int8 **)(v24 + 8LL * v20);
                if ( v22 != v21 )
                {
                  while ( v21 != (unsigned __int8 *)-4096LL )
                  {
                    v16 = (unsigned int)(v15 + 1);
                    v20 = (v23 - 1) & (v15 + v20);
                    v21 = *(unsigned __int8 **)(v24 + 8LL * v20);
                    if ( v22 == v21 )
                      goto LABEL_19;
                    v15 = (unsigned int)v16;
                  }
                  if ( v18 == v19 )
                  {
LABEL_55:
                    v29 = v18;
LABEL_56:
                    v77 = v18;
                    sub_2B49BC0(a2, *v19);
                    v18 = v77;
                    goto LABEL_57;
                  }
                  goto LABEL_24;
                }
LABEL_19:
                ++v19;
LABEL_20:
                LODWORD(v99) = 0;
                if ( v18 == v19 )
                  goto LABEL_44;
              }
              else
              {
                if ( v18 == v19 )
                  goto LABEL_55;
LABEL_24:
                for ( i = v19 + 1; ; v22 = *(i - 1) )
                {
                  v29 = i - 1;
                  if ( v23 )
                  {
                    v26 = v23 - 1;
                    v27 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                    v28 = *(unsigned __int8 **)(v24 + 8LL * v27);
                    if ( v28 == v22 )
                    {
                      v29 = i;
                      goto LABEL_27;
                    }
                    v15 = *(_QWORD *)(v24 + 8LL * (v26 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4))));
                    v57 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                    v16 = 1;
                    while ( v15 != -4096 )
                    {
                      v57 = v26 & (v16 + v57);
                      v15 = *(_QWORD *)(v24 + 8LL * v57);
                      if ( (unsigned __int8 *)v15 == v22 )
                      {
                        v29 = i;
                        if ( !v22 )
                          goto LABEL_27;
                        goto LABEL_89;
                      }
                      v16 = (unsigned int)(v16 + 1);
                    }
                  }
                  v30 = *v19;
                  if ( *v19 != v22 )
                    break;
LABEL_81:
                  v29 = i;
                  if ( !v22 )
                    goto LABEL_27;
                  v23 = *(_DWORD *)(a2 + 2000);
                  v24 = *(_QWORD *)(a2 + 1984);
                  if ( !v23 )
                    goto LABEL_71;
LABEL_83:
                  v26 = v23 - 1;
                  v27 = (v23 - 1) & (((unsigned int)v22 >> 4) ^ ((unsigned int)v22 >> 9));
                  v28 = *(unsigned __int8 **)(v24 + 8LL * v27);
                  if ( v28 == v22 )
                    goto LABEL_27;
LABEL_89:
                  v58 = 1;
                  while ( v28 != (unsigned __int8 *)-4096LL )
                  {
                    v15 = (unsigned int)(v58 + 1);
                    v27 = v26 & (v58 + v27);
                    v28 = *(unsigned __int8 **)(v24 + 8LL * v27);
                    if ( v28 == v22 )
                      goto LABEL_27;
                    ++v58;
                  }
LABEL_71:
                  v51 = (unsigned int)v99;
                  v52 = (unsigned int)v99 + 1LL;
                  if ( v52 > HIDWORD(v99) )
                  {
                    v80 = v18;
                    sub_C8D5F0((__int64)&v98, v100, v52, 8u, v15, v16);
                    v51 = (unsigned int)v99;
                    v18 = v80;
                  }
                  v98[v51] = (__int64)v22;
                  LODWORD(v99) = v99 + 1;
LABEL_27:
                  if ( v18 == i )
                    goto LABEL_32;
                  v24 = *(_QWORD *)(a2 + 1984);
                  v23 = *(_DWORD *)(a2 + 2000);
                  ++i;
                }
                v31 = (unsigned __int8 *)*((_QWORD *)v30 - 8);
                v32 = (unsigned __int8 *)*((_QWORD *)v22 - 8);
                if ( *((_QWORD *)v32 + 1) != *((_QWORD *)v31 + 1)
                  || *(_QWORD *)(*((_QWORD *)v22 - 4) + 8LL) != *(_QWORD *)(*((_QWORD *)v30 - 4) + 8LL) )
                {
                  goto LABEL_32;
                }
                v15 = *v32;
                v16 = (unsigned int)(v15 - 12);
                if ( (unsigned int)v16 <= 1 )
                  goto LABEL_70;
                v53 = *v31;
                v16 = (unsigned int)(v53 - 12);
                v78 = *v31;
                if ( (unsigned int)v16 <= 1 )
                  goto LABEL_70;
                if ( (unsigned __int8)v15 > 0x1Cu && v78 > 0x1Cu )
                {
                  if ( *((_QWORD *)v32 + 5) != *((_QWORD *)v31 + 5) )
                    goto LABEL_32;
                  v87[1] = *((_QWORD *)v30 - 8);
                  v79 = v18;
                  v54 = *(__int64 **)(a1 + 16);
                  v87[0] = (__int64)v32;
                  v55 = sub_2B5F980(v87, 2u, v54);
                  v18 = v79;
                  if ( !v55 || !v56 )
                    goto LABEL_32;
                  v22 = *(i - 1);
                  goto LABEL_81;
                }
                if ( v78 <= 0x15u && *v32 <= 0x15u || (_DWORD)v15 == v53 )
                {
LABEL_70:
                  v29 = i;
                  if ( !v23 )
                    goto LABEL_71;
                  goto LABEL_83;
                }
LABEL_32:
                v33 = v99;
                if ( (unsigned int)v99 <= 1 )
                  goto LABEL_56;
                v75 = v18;
                v34 = sub_2BCC8E0((__int64 *)a1, v98, (unsigned int)v99, a2, (__int64)&v88, v16, a3);
                if ( v34 )
                {
                  v83 = v34;
                  sub_2B3B230((__int64)&v98, (__int64)&v95, v35, v36, v37, v38);
                  v59 = v98;
                  LODWORD(v96) = 0;
                  v15 = v83;
                  v18 = v75;
                  v60 = &v98[(unsigned int)v99];
                  if ( v98 == v60 )
                  {
                    v19 = v29;
                    goto LABEL_20;
                  }
                  v41 = 0;
                  while ( 2 )
                  {
                    while ( 1 )
                    {
                      v64 = *v59;
                      if ( !*v59 )
                        break;
                      v65 = *(_DWORD *)(a2 + 2000);
                      v66 = *(_QWORD *)(a2 + 1984);
                      if ( v65 )
                      {
                        v61 = v65 - 1;
                        v62 = v61 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                        v63 = *(_QWORD *)(v66 + 8LL * v62);
                        if ( v64 == v63 )
                          break;
                        v16 = 1;
                        while ( v63 != -4096 )
                        {
                          v62 = v61 & (v16 + v62);
                          v63 = *(_QWORD *)(v66 + 8LL * v62);
                          if ( v64 == v63 )
                            goto LABEL_98;
                          v16 = (unsigned int)(v16 + 1);
                        }
                      }
                      if ( v41 + 1 > (unsigned __int64)HIDWORD(v96) )
                      {
                        v72 = v15;
                        v82 = v60;
                        v84 = v18;
                        sub_C8D5F0((__int64)&v95, v97, v41 + 1, 8u, v15, v16);
                        v41 = (unsigned int)v96;
                        v15 = v72;
                        v60 = v82;
                        v18 = v84;
                      }
                      ++v59;
                      v95[v41] = v64;
                      v41 = (unsigned int)(v96 + 1);
                      LODWORD(v96) = v96 + 1;
                      if ( v60 == v59 )
                      {
LABEL_104:
                        v83 = v15;
                        goto LABEL_38;
                      }
                    }
LABEL_98:
                    if ( v60 == ++v59 )
                      goto LABEL_104;
                    continue;
                  }
                }
                v39 = sub_2B49BC0(a2, *v19);
                v18 = v75;
                v15 = v39;
                v40 = *(_DWORD *)(a2 + 3360) / v39;
                if ( v40 < 2 )
                  v40 = 2;
                if ( v40 <= v33 )
                {
                  LODWORD(v41) = v96;
                  goto LABEL_38;
                }
LABEL_57:
                v41 = (unsigned int)v96;
                if ( !(_DWORD)v96 || *(_QWORD *)(*v95 + 8) == *((_QWORD *)*v19 + 1) )
                {
                  v44 = v98;
                  v15 = (__int64)&v98[(unsigned int)v99];
                  if ( v98 != (__int64 *)v15 )
                  {
                    while ( 1 )
                    {
                      v48 = *v44;
                      if ( *v44 )
                      {
                        v49 = *(_DWORD *)(a2 + 2000);
                        v50 = *(_QWORD *)(a2 + 1984);
                        if ( !v49 )
                          goto LABEL_65;
                        v45 = v49 - 1;
                        v46 = (v49 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                        v47 = *(_QWORD *)(v50 + 8LL * v46);
                        if ( v48 != v47 )
                          break;
                      }
LABEL_62:
                      if ( (__int64 *)v15 == ++v44 )
                        goto LABEL_38;
                    }
                    v16 = 1;
                    while ( v47 != -4096 )
                    {
                      v46 = v45 & (v16 + v46);
                      v47 = *(_QWORD *)(v50 + 8LL * v46);
                      if ( v48 == v47 )
                        goto LABEL_62;
                      v16 = (unsigned int)(v16 + 1);
                    }
LABEL_65:
                    if ( v41 + 1 > (unsigned __int64)HIDWORD(v96) )
                    {
                      v71 = v15;
                      v81 = v18;
                      sub_C8D5F0((__int64)&v95, v97, v41 + 1, 8u, v15, v16);
                      v41 = (unsigned int)v96;
                      v15 = v71;
                      v18 = v81;
                    }
                    v95[v41] = v48;
                    v41 = (unsigned int)(v96 + 1);
                    LODWORD(v96) = v96 + 1;
                    goto LABEL_62;
                  }
                }
LABEL_38:
                if ( (unsigned int)v41 <= 1 )
                {
                  v19 = v29;
                  goto LABEL_20;
                }
                if ( v18 != v29 && *((_QWORD *)*v29 + 1) == *((_QWORD *)*v19 + 1) )
                {
                  LODWORD(v99) = 0;
                  v19 = v29;
                }
                else
                {
                  v19 = v29;
                  v76 = v18;
                  v42 = sub_2BCC8E0((__int64 *)a1, v95, (unsigned int)v41, a2, (__int64)&v88, v16, a3);
                  v43 = v83;
                  v18 = v76;
                  LODWORD(v96) = 0;
                  LODWORD(v99) = 0;
                  if ( v42 )
                    v43 = v42;
                  v83 = v43;
                  if ( v76 == v29 )
                  {
LABEL_44:
                    v70 |= v83;
                    if ( v98 != (__int64 *)v100 )
                      _libc_free((unsigned __int64)v98);
                    if ( v95 != (__int64 *)v97 )
                      _libc_free((unsigned __int64)v95);
                    v17 = v92;
                    break;
                  }
                }
              }
            }
          }
          if ( v17 != v94 )
            _libc_free((unsigned __int64)v17);
        }
      }
      v85 += 88;
    }
    while ( v73 != v85 );
    v67 = v89;
    v68 = 40LL * (unsigned int)v91;
  }
  sub_C7D6A0(v67, v68, 8);
  return v70;
}
