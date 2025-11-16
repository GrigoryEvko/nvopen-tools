// Function: sub_3088A20
// Address: 0x3088a20
//
void __fastcall sub_3088A20(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // rcx
  __int64 *v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  __int64 v25; // rdx
  unsigned __int64 *v26; // r15
  __int64 v27; // r13
  unsigned __int64 *v28; // r12
  __int64 i; // r15
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // r15
  __int64 v38; // rbx
  _BYTE *v39; // rax
  char v40; // al
  __int64 v41; // rax
  _BYTE *v42; // rsi
  unsigned __int64 v43; // rdi
  __int64 v44; // r14
  _QWORD *v45; // rax
  __int64 *v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // r14
  unsigned __int8 v50; // al
  unsigned __int64 *v51; // rdx
  unsigned __int64 v52; // rax
  unsigned __int64 *v53; // rsi
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r14
  unsigned __int8 v58; // al
  unsigned __int8 v59; // r9
  unsigned __int64 v60; // rax
  unsigned __int64 *v61; // rsi
  unsigned __int64 *v62; // rsi
  void *v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rcx
  unsigned __int64 *v68; // r14
  unsigned __int64 *v69; // r13
  unsigned __int64 v70; // r8
  __int64 v71; // r14
  unsigned __int64 *v72; // r13
  __int64 j; // r14
  unsigned __int64 v74; // rdi
  __int64 v75; // r14
  _QWORD *v76; // rax
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  _BYTE *v79; // rdx
  __int64 v80; // [rsp+0h] [rbp-320h]
  __int64 v81; // [rsp+18h] [rbp-308h]
  __int64 v83; // [rsp+28h] [rbp-2F8h]
  unsigned __int64 v84; // [rsp+38h] [rbp-2E8h] BYREF
  _QWORD v85[2]; // [rsp+40h] [rbp-2E0h] BYREF
  unsigned __int64 v86; // [rsp+50h] [rbp-2D0h] BYREF
  _BYTE *v87; // [rsp+58h] [rbp-2C8h]
  _BYTE *v88; // [rsp+60h] [rbp-2C0h]
  unsigned __int64 *v89; // [rsp+70h] [rbp-2B0h] BYREF
  unsigned __int64 *v90; // [rsp+78h] [rbp-2A8h]
  unsigned __int64 *v91; // [rsp+80h] [rbp-2A0h]
  unsigned __int64 *v92; // [rsp+90h] [rbp-290h] BYREF
  unsigned __int64 *v93; // [rsp+98h] [rbp-288h] BYREF
  unsigned __int64 *v94; // [rsp+A0h] [rbp-280h]
  unsigned __int64 **v95; // [rsp+A8h] [rbp-278h]
  unsigned __int64 **v96; // [rsp+B0h] [rbp-270h]
  __int64 v97; // [rsp+B8h] [rbp-268h]
  __int64 *v98; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v99; // [rsp+C8h] [rbp-258h]
  _QWORD v100[32]; // [rsp+D0h] [rbp-250h] BYREF
  __int64 v101; // [rsp+1D0h] [rbp-150h] BYREF
  void *s; // [rsp+1D8h] [rbp-148h] BYREF
  __int128 v103; // [rsp+1E0h] [rbp-140h]
  void **p_s; // [rsp+1F0h] [rbp-130h] BYREF
  __int64 v105; // [rsp+1F8h] [rbp-128h]

  v2 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 40) + 312LL;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_201:
    BUG();
  v5 = a1;
  while ( *(_UNKNOWN **)v3 != &unk_4F89C28 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_201;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F89C28);
  v81 = sub_DFED00(v6, a2);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, v7, v8);
    v9 = *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      sub_B2C6D0(a2, a2, v66, v67);
    v10 = *(_QWORD *)(a2 + 96);
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 96);
    v10 = v9;
  }
  v11 = v10 + 40LL * *(_QWORD *)(a2 + 104);
  if ( v11 != v9 )
  {
    while ( 1 )
    {
      while ( *(_BYTE *)(*(_QWORD *)(v9 + 8) + 8LL) != 14 || !(unsigned __int8)sub_B2D700(v9) )
      {
LABEL_9:
        v9 += 40;
        if ( v9 == v11 )
          goto LABEL_49;
      }
      v101 = 0;
      v98 = v100;
      v99 = 0x2000000000LL;
      s = &p_s;
      *(_QWORD *)&v103 = 32;
      DWORD2(v103) = 0;
      BYTE12(v103) = 1;
      v89 = 0;
      v90 = 0;
      v91 = 0;
      v14 = *(_QWORD *)(v9 + 16);
      if ( !v14 )
      {
        v21 = (__int64)v100;
        v20 = 0;
        goto LABEL_25;
      }
LABEL_13:
      v15 = s;
      v16 = DWORD1(v103);
      v17 = (__int64 *)((char *)s + 8 * DWORD1(v103));
      if ( s == v17 )
      {
LABEL_22:
        if ( DWORD1(v103) < (unsigned int)v103 )
        {
          ++DWORD1(v103);
          *v17 = v14;
          ++v101;
          goto LABEL_17;
        }
        goto LABEL_21;
      }
      while ( *v15 != v14 )
      {
        if ( v17 == ++v15 )
          goto LABEL_22;
      }
      while ( 1 )
      {
LABEL_17:
        v18 = (unsigned int)v99;
        v16 = HIDWORD(v99);
        v19 = (unsigned int)v99 + 1LL;
        if ( v19 > HIDWORD(v99) )
        {
          sub_C8D5F0((__int64)&v98, v100, v19, 8u, v12, v13);
          v18 = (unsigned int)v99;
        }
        v17 = v98;
        v98[v18] = v14;
        v20 = v99 + 1;
        LODWORD(v99) = v99 + 1;
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
          break;
        if ( BYTE12(v103) )
          goto LABEL_13;
LABEL_21:
        sub_C8CC70((__int64)&v101, v14, (__int64)v17, v16, v12, v13);
      }
      v21 = (__int64)v98;
LABEL_25:
      v22 = v21 + 8LL * v20;
      while ( 2 )
      {
        if ( v20 )
        {
          v23 = *(_QWORD *)(v22 - 8);
          LODWORD(v99) = --v20;
          v24 = *(unsigned __int8 **)(v23 + 24);
          v25 = (unsigned int)*v24 - 29;
          switch ( *v24 )
          {
            case 0x1Eu:
            case 0x52u:
              v22 -= 8LL;
              continue;
            case 0x22u:
            case 0x55u:
              if ( (unsigned int)sub_B49240((__int64)v24) == 8170 )
              {
                v75 = *((_QWORD *)v24 + 2);
                if ( v75 )
                {
                  if ( BYTE12(v103) )
                  {
LABEL_174:
                    v76 = s;
                    v47 = DWORD1(v103);
                    v46 = (__int64 *)((char *)s + 8 * DWORD1(v103));
                    if ( s == v46 )
                      goto LABEL_185;
                    while ( *v76 != v75 )
                    {
                      if ( v46 == ++v76 )
                      {
LABEL_185:
                        if ( DWORD1(v103) < (unsigned int)v103 )
                        {
                          ++DWORD1(v103);
                          *v46 = v75;
                          ++v101;
                          goto LABEL_181;
                        }
                        goto LABEL_180;
                      }
                    }
                    goto LABEL_178;
                  }
                  while ( 1 )
                  {
LABEL_180:
                    sub_C8CC70((__int64)&v101, v75, (__int64)v46, v47, v12, v13);
                    if ( (_BYTE)v46 )
                    {
LABEL_181:
                      v77 = (unsigned int)v99;
                      v47 = HIDWORD(v99);
                      v78 = (unsigned int)v99 + 1LL;
                      if ( v78 > HIDWORD(v99) )
                      {
                        sub_C8D5F0((__int64)&v98, v100, v78, 8u, v12, v13);
                        v77 = (unsigned int)v99;
                      }
                      v46 = v98;
                      v98[v77] = v75;
                      LODWORD(v99) = v99 + 1;
                      v75 = *(_QWORD *)(v75 + 8);
                      if ( !v75 )
                        break;
                    }
                    else
                    {
LABEL_178:
                      v75 = *(_QWORD *)(v75 + 8);
                      if ( !v75 )
                        break;
                    }
                    if ( BYTE12(v103) )
                      goto LABEL_174;
                  }
                }
LABEL_89:
                v21 = (__int64)v98;
                v20 = v99;
              }
              else
              {
                if ( sub_B49E00((__int64)v24) )
                  goto LABEL_89;
                if ( !(unsigned __int8)sub_B49E20((__int64)v24) )
                  goto LABEL_35;
                v21 = (__int64)v98;
                v20 = v99;
              }
              goto LABEL_25;
            case 0x3Du:
              v85[0] = v24;
              if ( (v24[2] & 1) != 0 )
                goto LABEL_25;
              v48 = *(_QWORD *)(*((_QWORD *)v24 - 4) + 8LL);
              if ( *(_BYTE *)(v48 + 8) != 14 || *(_DWORD *)(v48 + 8) >> 8 != 1 )
                goto LABEL_25;
              v49 = *((_QWORD *)v24 + 1);
              v50 = *(_BYTE *)(v49 + 8);
              if ( v50 != 12 )
                goto LABEL_99;
              v86 = sub_BCAE30(v49);
              v87 = v79;
              if ( (unsigned __int64)sub_CA1930(&v86) <= 0x40 )
                goto LABEL_107;
              v50 = *(_BYTE *)(v49 + 8);
LABEL_99:
              if ( v50 <= 3u || v50 == 5 || (v50 & 0xFD) == 4 || v50 == 15 )
                goto LABEL_107;
              if ( v50 != 17 )
                goto LABEL_106;
              v92 = (unsigned __int64 *)sub_BCAE30(*(_QWORD *)(v49 + 24));
              v93 = v51;
              if ( (unsigned int)sub_CA1930(&v92) > 7 )
                goto LABEL_107;
              v50 = *(_BYTE *)(v49 + 8);
LABEL_106:
              if ( v50 == 14 )
              {
LABEL_107:
                v12 = (unsigned int)sub_AE5020(*(_QWORD *)(v5 + 232), v49);
                _BitScanReverse64(&v52, 1LL << (*(_WORD *)(v85[0] + 2LL) >> 1));
                if ( (unsigned __int8)v12 <= (unsigned __int8)(63 - (v52 ^ 0x3F)) )
                {
                  v53 = v90;
                  if ( v90 == v91 )
                  {
                    sub_27D05B0((__int64)&v89, v90, v85);
                  }
                  else
                  {
                    if ( v90 )
                    {
                      *v90 = v85[0];
                      v53 = v90;
                    }
                    v90 = v53 + 1;
                  }
                }
              }
              goto LABEL_89;
            case 0x3Fu:
            case 0x4Eu:
            case 0x4Fu:
            case 0x54u:
            case 0x56u:
              v44 = *((_QWORD *)v24 + 2);
              if ( !v44 )
                goto LABEL_25;
              break;
            default:
              goto LABEL_35;
          }
          while ( 1 )
          {
            if ( !BYTE12(v103) )
              goto LABEL_112;
            v45 = s;
            v21 = DWORD1(v103);
            v25 = (__int64)s + 8 * DWORD1(v103);
            if ( s != (void *)v25 )
            {
              while ( *v45 != v44 )
              {
                if ( (_QWORD *)v25 == ++v45 )
                  goto LABEL_117;
              }
LABEL_88:
              v44 = *(_QWORD *)(v44 + 8);
              if ( !v44 )
                goto LABEL_89;
              continue;
            }
LABEL_117:
            if ( DWORD1(v103) < (unsigned int)v103 )
            {
              ++DWORD1(v103);
              *(_QWORD *)v25 = v44;
              ++v101;
            }
            else
            {
LABEL_112:
              sub_C8CC70((__int64)&v101, v44, v25, v21, v12, v13);
              if ( !(_BYTE)v25 )
                goto LABEL_88;
            }
            v54 = (unsigned int)v99;
            v21 = HIDWORD(v99);
            v55 = (unsigned int)v99 + 1LL;
            if ( v55 > HIDWORD(v99) )
            {
              sub_C8D5F0((__int64)&v98, v100, v55, 8u, v12, v13);
              v54 = (unsigned int)v99;
            }
            v25 = (__int64)v98;
            v98[v54] = v44;
            LODWORD(v99) = v99 + 1;
            v44 = *(_QWORD *)(v44 + 8);
            if ( !v44 )
              goto LABEL_89;
          }
        }
        break;
      }
      LODWORD(v93) = 0;
      v95 = &v93;
      v96 = &v93;
      v94 = 0;
      v97 = 0;
      if ( v89 == v90 )
      {
        v30 = 0;
      }
      else
      {
        v26 = v90;
        v80 = v11;
        v27 = v5;
        v28 = v89;
        do
        {
          v22 = *v28++;
          sub_3086F90(*(_QWORD *)(v27 + 232), v22, (__int64)&v92, v81);
        }
        while ( v26 != v28 );
        v5 = v27;
        v11 = v80;
        for ( i = (__int64)v95; (unsigned __int64 **)i != &v93; i = sub_220EF30(i) )
          sub_B43D60(*(_QWORD **)(i + 32));
        v30 = (unsigned __int64)v94;
      }
      sub_3085130(v30);
      v94 = 0;
      v97 = 0;
      v95 = &v93;
      v96 = &v93;
      sub_3085130(0);
LABEL_35:
      v31 = (unsigned __int64)v89;
      if ( v89 != v90 )
        v90 = v89;
      ++v101;
      LODWORD(v99) = 0;
      if ( BYTE12(v103) )
        goto LABEL_42;
      v32 = 4 * (DWORD1(v103) - DWORD2(v103));
      if ( v32 < 0x20 )
        v32 = 32;
      if ( (unsigned int)v103 <= v32 )
        break;
      sub_C8C990((__int64)&v101, v22);
      v31 = (unsigned __int64)v89;
LABEL_43:
      if ( v31 )
        j_j___libc_free_0(v31);
      if ( !BYTE12(v103) )
        _libc_free((unsigned __int64)s);
      if ( v98 == v100 )
        goto LABEL_9;
      _libc_free((unsigned __int64)v98);
      v9 += 40;
      if ( v9 == v11 )
        goto LABEL_49;
    }
    memset(s, -1, 8LL * (unsigned int)v103);
    v31 = (unsigned __int64)v89;
LABEL_42:
    *(_QWORD *)((char *)&v103 + 4) = 0;
    goto LABEL_43;
  }
LABEL_49:
  v86 = 0;
  v33 = *(__int64 **)(v5 + 8);
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v98 = 0;
  v99 = 0;
  v100[0] = 0;
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
LABEL_204:
    BUG();
  while ( *(_UNKNOWN **)v34 != &unk_4F881C8 )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_204;
  }
  *(_QWORD *)(v5 + 224) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(
                                        *(_QWORD *)(v34 + 8),
                                        &unk_4F881C8)
                                    + 176);
  v36 = *(_QWORD *)(a2 + 80);
  v83 = a2 + 72;
  if ( v36 != v83 )
  {
    while ( 1 )
    {
      if ( !v36 )
        BUG();
      v37 = *(_QWORD *)(v36 + 32);
      v38 = v36 + 24;
      if ( v37 != v36 + 24 )
        break;
LABEL_71:
      v36 = *(_QWORD *)(v36 + 8);
      if ( v83 == v36 )
        goto LABEL_72;
    }
    while ( 1 )
    {
      if ( !v37 )
        BUG();
      if ( *(_BYTE *)(v37 - 24) == 85 )
      {
        v85[0] = v37 - 24;
        if ( (unsigned __int8)sub_B46490(v37 - 24) )
        {
          v39 = *(_BYTE **)(v37 - 56);
          if ( *v39 != 25 || v39[96] )
          {
            v62 = v90;
            if ( v90 == v91 )
            {
              sub_2628C60((__int64)&v89, v90, v85);
            }
            else
            {
              if ( v90 )
              {
                *v90 = v37 - 24;
                v62 = v90;
              }
              v90 = v62 + 1;
            }
          }
        }
        goto LABEL_60;
      }
      v85[0] = 0;
      v101 = 0;
      v40 = *(_BYTE *)(v37 - 24);
      switch ( v40 )
      {
        case '>':
          v41 = *(_QWORD *)(v37 - 56);
          v101 = v41;
          break;
        case 'B':
          v41 = *(_QWORD *)(v37 - 88);
          v101 = v41;
          break;
        case 'A':
          v41 = *(_QWORD *)(v37 - 120);
          v101 = v41;
          break;
        default:
          goto LABEL_119;
      }
      if ( v41 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(v41 + 8) + 8LL) > 0x1FFu )
          goto LABEL_60;
        v42 = v87;
        if ( v87 == v88 )
        {
          sub_9281F0((__int64)&v86, v87, &v101);
          goto LABEL_60;
        }
        if ( v87 )
        {
          *(_QWORD *)v87 = v41;
          v42 = v87;
        }
        v87 = v42 + 8;
        v37 = *(_QWORD *)(v37 + 8);
        if ( v38 == v37 )
          goto LABEL_71;
      }
      else
      {
LABEL_119:
        if ( *(_BYTE *)(v37 - 24) == 61 )
        {
          v84 = v37 - 24;
          if ( (*(_BYTE *)(v37 - 22) & 1) == 0 )
          {
            v56 = *(_QWORD *)(*(_QWORD *)(v37 - 56) + 8LL);
            if ( *(_BYTE *)(v56 + 8) == 14 && *(_DWORD *)(v56 + 8) >> 8 == 1 )
            {
              v57 = *(_QWORD *)(v37 - 16);
              v58 = *(_BYTE *)(v57 + 8);
              if ( v58 == 12 )
              {
                v85[0] = sub_BCAE30(*(_QWORD *)(v37 - 16));
                v85[1] = v65;
                if ( (unsigned __int64)sub_CA1930(v85) <= 0x40 )
                  goto LABEL_128;
                v58 = *(_BYTE *)(v57 + 8);
              }
              if ( v58 <= 3u || v58 == 5 || (v58 & 0xFD) == 4 || v58 == 15 )
                goto LABEL_128;
              if ( v58 == 17 )
              {
                v101 = sub_BCAE30(*(_QWORD *)(v57 + 24));
                s = v63;
                if ( (unsigned int)sub_CA1930(&v101) > 7 )
                  goto LABEL_128;
                v58 = *(_BYTE *)(v57 + 8);
              }
              if ( v58 == 14 )
              {
LABEL_128:
                v59 = sub_AE5020(*(_QWORD *)(v5 + 232), v57);
                _BitScanReverse64(&v60, 1LL << (*(_WORD *)(v84 + 2) >> 1));
                if ( v59 <= (unsigned __int8)(63 - (v60 ^ 0x3F)) )
                {
                  if ( (unsigned __int8)sub_3088980((_QWORD *)(v5 + 176), v84, *(_QWORD *)(v5 + 224)) )
                  {
                    v61 = v93;
                    if ( v93 == v94 )
                    {
                      sub_27D05B0((__int64)&v92, v93, &v84);
                    }
                    else
                    {
                      if ( v93 )
                      {
                        *v93 = v84;
                        v61 = v93;
                      }
                      v93 = v61 + 1;
                    }
                  }
                  else
                  {
                    v64 = v99;
                    if ( v99 == v100[0] )
                    {
                      sub_27D05B0((__int64)&v98, (_BYTE *)v99, &v84);
                    }
                    else
                    {
                      if ( v99 )
                      {
                        *(_QWORD *)v99 = v84;
                        v64 = v99;
                      }
                      v99 = v64 + 8;
                    }
                  }
                }
              }
            }
          }
        }
LABEL_60:
        v37 = *(_QWORD *)(v37 + 8);
        if ( v38 == v37 )
          goto LABEL_71;
      }
    }
  }
LABEL_72:
  if ( v90 == v89 )
  {
    v68 = v93;
    LODWORD(s) = 0;
    *(_QWORD *)&v103 = 0;
    *((_QWORD *)&v103 + 1) = &s;
    p_s = &s;
    v105 = 0;
    if ( v92 == v93 )
    {
      v70 = (unsigned __int64)v98;
      v71 = v99;
      v74 = 0;
      if ( (__int64 *)v99 == v98 )
      {
LABEL_171:
        sub_3085130(v74);
        *(_QWORD *)&v103 = 0;
        *((_QWORD *)&v103 + 1) = &s;
        p_s = &s;
        v105 = 0;
        sub_3085130(0);
        goto LABEL_73;
      }
    }
    else
    {
      v69 = v92;
      do
      {
        if ( !(unsigned __int8)sub_30888F0(v5, (__int64 *)&v86, *v69) )
          sub_3086F90(*(_QWORD *)(v5 + 232), *v69, (__int64)&v101, v81);
        ++v69;
      }
      while ( v68 != v69 );
      v70 = (unsigned __int64)v98;
      v71 = v99;
      if ( v98 == (__int64 *)v99 )
        goto LABEL_168;
    }
    v72 = (unsigned __int64 *)v70;
    do
    {
      if ( !(unsigned __int8)sub_30888F0(v5, (__int64 *)&v86, *v72) )
        sub_3086F90(*(_QWORD *)(v5 + 232), *v72, (__int64)&v101, v81);
      ++v72;
    }
    while ( (unsigned __int64 *)v71 != v72 );
LABEL_168:
    for ( j = *((_QWORD *)&v103 + 1); (void **)j != &s; j = sub_220EF30(j) )
      sub_B43D60(*(_QWORD **)(j + 32));
    v74 = v103;
    goto LABEL_171;
  }
LABEL_73:
  sub_30854D0(*(_QWORD **)(v5 + 192));
  v43 = (unsigned __int64)v98;
  *(_QWORD *)(v5 + 192) = 0;
  *(_QWORD *)(v5 + 200) = v5 + 184;
  *(_QWORD *)(v5 + 208) = v5 + 184;
  *(_QWORD *)(v5 + 216) = 0;
  if ( v43 )
    j_j___libc_free_0(v43);
  if ( v92 )
    j_j___libc_free_0((unsigned __int64)v92);
  if ( v89 )
    j_j___libc_free_0((unsigned __int64)v89);
  if ( v86 )
    j_j___libc_free_0(v86);
}
