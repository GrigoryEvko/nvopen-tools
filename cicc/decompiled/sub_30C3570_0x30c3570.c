// Function: sub_30C3570
// Address: 0x30c3570
//
_QWORD *__fastcall sub_30C3570(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // rcx
  unsigned __int64 v4; // rax
  char v7; // si
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *i; // r12
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rcx
  __int64 v16; // r13
  _QWORD *result; // rax
  bool v18; // al
  unsigned __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rdx
  char v22; // si
  __int64 v23; // rax
  int v24; // esi
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // rsi
  char v31; // al
  __int64 v32; // rax
  unsigned __int64 v33; // r9
  _BYTE *j; // rdi
  __int64 v35; // rdx
  unsigned __int8 v36; // al
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rdx
  unsigned __int8 v41; // al
  int v42; // edx
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  int v47; // r13d
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  int v51; // edx
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r13
  __int64 v58; // rdx
  __int64 v59; // r8
  unsigned __int8 v60; // dl
  int v61; // edx
  unsigned __int64 v62; // rax
  __int64 v63; // r12
  int v64; // eax
  unsigned int v65; // r13d
  __int64 v66; // rax
  __int64 v67; // rdx
  char v68; // di
  __int64 v69; // rax
  int v70; // edi
  int v71; // [rsp+Ch] [rbp-144h]
  unsigned int v72; // [rsp+10h] [rbp-140h]
  int v73; // [rsp+10h] [rbp-140h]
  _QWORD *v74; // [rsp+18h] [rbp-138h]
  _QWORD *v75; // [rsp+18h] [rbp-138h]
  __int64 v76; // [rsp+18h] [rbp-138h]
  __int64 v77; // [rsp+18h] [rbp-138h]
  _QWORD *v78; // [rsp+18h] [rbp-138h]
  _QWORD *v79; // [rsp+18h] [rbp-138h]
  unsigned __int64 v80; // [rsp+20h] [rbp-130h]
  unsigned __int64 v81; // [rsp+30h] [rbp-120h]
  _BYTE v82[16]; // [rsp+40h] [rbp-110h] BYREF
  void (__fastcall *v83)(_BYTE *, _BYTE *, __int64); // [rsp+50h] [rbp-100h]
  unsigned __int8 (__fastcall *v84)(_BYTE *, unsigned __int64, __int64, __int64, unsigned __int64, unsigned __int64); // [rsp+58h] [rbp-F8h]
  __m128i v85; // [rsp+60h] [rbp-F0h]
  __m128i v86; // [rsp+70h] [rbp-E0h]
  _BYTE v87[16]; // [rsp+80h] [rbp-D0h] BYREF
  void (__fastcall *v88)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-C0h]
  __int64 v89; // [rsp+98h] [rbp-B8h]
  __m128i v90; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v91; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE v92[16]; // [rsp+C0h] [rbp-90h] BYREF
  void (__fastcall *v93)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-80h]
  unsigned __int8 (__fastcall *v94)(_BYTE *, unsigned __int64, __int64, __int64, unsigned __int64, unsigned __int64); // [rsp+D8h] [rbp-78h]
  __m128i v95; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v96; // [rsp+F0h] [rbp-60h] BYREF
  _BYTE v97[16]; // [rsp+100h] [rbp-50h] BYREF
  void (__fastcall *v98)(_BYTE *, _BYTE *, __int64); // [rsp+110h] [rbp-40h]
  __int64 v99; // [rsp+118h] [rbp-38h]

  v3 = a2 + 6;
  *a1 += a3;
  v4 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 6 == (_QWORD *)v4 || !v4 || (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_198:
    BUG();
  v7 = *(_BYTE *)(v4 - 24);
  if ( v7 == 31 )
  {
    v9 = 2 * a3;
    if ( (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) != 3 )
      v9 = 0;
  }
  else
  {
    v9 = 0;
    if ( v7 == 32 )
      v9 = a3 * (((*(_DWORD *)(v4 - 20) & 0x7FFFFFFu) >> 1) - (*(_QWORD *)(*(_QWORD *)(v4 - 32) + 32LL) == 0));
  }
  v10 = 0x8000000000041LL;
  a1[1] += v9;
  for ( i = (_QWORD *)a2[7]; i != v3; i = (_QWORD *)i[1] )
  {
    while ( 1 )
    {
      if ( !i )
        goto LABEL_198;
      v12 = *((unsigned __int8 *)i - 24);
      if ( (unsigned __int8)(v12 - 34) <= 0x33u )
        break;
LABEL_10:
      i = (_QWORD *)i[1];
      if ( i == v3 )
        goto LABEL_20;
    }
    if ( _bittest64(&v10, (unsigned int)(v12 - 34)) )
    {
      v13 = *(i - 7);
      if ( v13 )
      {
        if ( !*(_BYTE *)v13 && i[7] == *(_QWORD *)(v13 + 24) && (*(_BYTE *)(v13 + 33) & 0x20) == 0 )
        {
          v75 = v3;
          v18 = sub_B2FC80(v13);
          v3 = v75;
          if ( !v18 )
            a1[3] += a3;
          LOBYTE(v12) = *((_BYTE *)i - 24);
        }
      }
    }
    if ( (_BYTE)v12 == 61 )
    {
      a1[4] += a3;
      goto LABEL_10;
    }
    if ( (_BYTE)v12 != 62 )
      goto LABEL_10;
    a1[5] += a3;
  }
LABEL_20:
  v74 = v3;
  v14 = sub_AA6A60((__int64)a2);
  v15 = v74;
  v16 = a3 * v14 + a1[8];
  result = &qword_502F040;
  a1[8] = v16;
  if ( !LOBYTE(qword_502F088[8]) )
    return result;
  v19 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v74 == (_QWORD *)v19 )
    goto LABEL_194;
  if ( !v19 )
    goto LABEL_193;
  if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
    goto LABEL_194;
  v20 = sub_B46E30(v19 - 24);
  v15 = v74;
  v72 = v20;
  if ( v20 != 1 )
  {
    if ( v20 == 2 )
    {
      a1[10] += a3;
      goto LABEL_33;
    }
    if ( v20 > 2 )
    {
      a1[11] += a3;
      goto LABEL_33;
    }
LABEL_194:
    v72 = 0;
    goto LABEL_33;
  }
  a1[9] += a3;
LABEL_33:
  v21 = a2[2];
  while ( v21 )
  {
    v22 = **(_BYTE **)(v21 + 24);
    v23 = v21;
    v21 = *(_QWORD *)(v21 + 8);
    if ( (unsigned __int8)(v22 - 30) <= 0xAu )
    {
      v24 = 0;
      while ( 1 )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v23 + 24) - 30) <= 0xAu )
        {
          v23 = *(_QWORD *)(v23 + 8);
          ++v24;
          if ( !v23 )
            goto LABEL_40;
        }
      }
LABEL_40:
      if ( v24 )
      {
        if ( v24 == 1 )
        {
          a1[13] += a3;
        }
        else if ( (unsigned int)(v24 + 1) > 2 )
        {
          a1[14] += a3;
        }
      }
      else
      {
        a1[12] += a3;
      }
      break;
    }
  }
  if ( v16 <= (unsigned int)qword_502EFE8 )
  {
    if ( v16 <= (unsigned int)qword_502EF08 )
      a1[17] += a3;
    else
      a1[16] += a3;
  }
  else
  {
    a1[15] += a3;
  }
  if ( v72 > 1 )
  {
    v62 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v15 != (_QWORD *)v62 )
    {
      if ( v62 )
      {
        v63 = v62 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v62 - 24) - 30 <= 0xA )
        {
          v78 = v15;
          v64 = sub_B46E30(v63);
          v15 = v78;
          v71 = v64;
          if ( v64 )
          {
            v65 = 0;
            do
            {
              v79 = v15;
              v66 = sub_B46EC0(v63, v65);
              v15 = v79;
              v67 = *(_QWORD *)(v66 + 16);
              if ( v67 )
              {
                while ( 1 )
                {
                  v68 = **(_BYTE **)(v67 + 24);
                  v69 = v67;
                  v67 = *(_QWORD *)(v67 + 8);
                  if ( (unsigned __int8)(v68 - 30) <= 0xAu )
                    break;
                  if ( !v67 )
                    goto LABEL_174;
                }
                v70 = 0;
                while ( 1 )
                {
                  v69 = *(_QWORD *)(v69 + 8);
                  if ( !v69 )
                    break;
                  while ( (unsigned __int8)(**(_BYTE **)(v69 + 24) - 30) <= 0xAu )
                  {
                    v69 = *(_QWORD *)(v69 + 8);
                    ++v70;
                    if ( !v69 )
                      goto LABEL_172;
                  }
                }
LABEL_172:
                if ( (unsigned int)(v70 + 1) > 1 )
                  a1[30] += a3;
              }
LABEL_174:
              ++v65;
            }
            while ( v71 != v65 );
          }
        }
        goto LABEL_47;
      }
LABEL_193:
      BUG();
    }
  }
LABEL_47:
  a1[31] += a3 * v72;
  v25 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 == (_QWORD *)v25 || !v25 || (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 > 0xA )
    goto LABEL_198;
  if ( *(_BYTE *)(v25 - 24) == 31 && (*(_DWORD *)(v25 - 20) & 0x7FFFFFF) != 3 )
    a1[32] += a3;
  sub_AA69B0(&v90, (__int64)a2, 1);
  v83 = 0;
  v80 = _mm_loadu_si128(&v90).m128i_u64[0];
  v81 = _mm_loadu_si128(&v91).m128i_u64[0];
  if ( v93 )
  {
    v93(v82, v92, 2);
    v84 = v94;
    v83 = v93;
  }
  v27 = _mm_loadu_si128(&v95);
  v28 = _mm_loadu_si128(&v96);
  v88 = 0;
  v85 = v27;
  v86 = v28;
  if ( v98 )
  {
    v98(v87, v97, 2);
    v89 = v99;
    v88 = v98;
  }
  while ( 1 )
  {
    v29 = v80;
    v30 = v80;
    if ( v85.m128i_i64[0] == v80 )
      break;
    while ( 1 )
    {
      if ( !v30 )
        goto LABEL_198;
      if ( (unsigned int)*(unsigned __int8 *)(v30 - 24) - 67 <= 0xC )
        a1[18] += a3;
      v31 = *(_BYTE *)(*(_QWORD *)(v30 - 16) + 8LL);
      if ( v31 == 2 )
      {
        a1[19] += a3;
        if ( *(_BYTE *)(v30 - 24) != 85 )
          goto LABEL_65;
      }
      else
      {
        if ( v31 == 12 )
          a1[20] += a3;
        if ( *(_BYTE *)(v30 - 24) != 85 )
        {
LABEL_65:
          v32 = *(_DWORD *)(v30 - 20) & 0x7FFFFFF;
          goto LABEL_66;
        }
      }
      v38 = *(_QWORD *)(v30 - 56);
      if ( v38 )
      {
        if ( !*(_BYTE *)v38 )
        {
          v26 = *(_QWORD *)(v30 + 56);
          if ( *(_QWORD *)(v38 + 24) == v26 && (*(_BYTE *)(v38 + 33) & 0x20) != 0 )
          {
            ++a1[33];
            if ( *(_BYTE *)(v30 - 24) != 85 )
              goto LABEL_65;
          }
        }
      }
      v39 = v30 - 24;
      if ( sub_B491E0(v30 - 24) )
      {
        a1[35] += a3;
        v40 = *(_QWORD *)(v30 - 16);
        v41 = *(_BYTE *)(v40 + 8);
        if ( v41 == 12 )
          goto LABEL_153;
      }
      else
      {
        a1[34] += a3;
        v40 = *(_QWORD *)(v30 - 16);
        v41 = *(_BYTE *)(v40 + 8);
        if ( v41 == 12 )
        {
LABEL_153:
          a1[36] += a3;
          v61 = *(unsigned __int8 *)(v30 - 24);
          v43 = v61 - 29;
          if ( v61 == 40 )
            goto LABEL_154;
          goto LABEL_112;
        }
      }
      if ( v41 <= 3u || v41 == 5 || (v41 & 0xFD) == 4 )
      {
        a1[37] += a3;
      }
      else if ( v41 == 14 )
      {
        a1[38] += a3;
      }
      else if ( v41 == 18 || v41 == 17 )
      {
        v59 = *(_QWORD *)(v40 + 16);
        v60 = *(_BYTE *)(*(_QWORD *)v59 + 8LL);
        if ( v60 == 12 )
        {
          a1[39] += a3;
        }
        else if ( v60 <= 3u || v60 == 5 || (v60 & 0xFD) == 4 )
        {
          a1[40] += a3;
        }
        else if ( (unsigned __int8)(v41 - 17) <= 1u && *(_BYTE *)(*(_QWORD *)v59 + 8LL) == 14 )
        {
          a1[41] += a3;
        }
      }
      v42 = *(unsigned __int8 *)(v30 - 24);
      v43 = v42 - 29;
      if ( v42 == 40 )
      {
LABEL_154:
        v76 = 32LL * (unsigned int)sub_B491D0(v39);
        goto LABEL_119;
      }
LABEL_112:
      v76 = 0;
      if ( v43 != 56 )
      {
        if ( v43 != 5 )
          goto LABEL_197;
        v76 = 64;
      }
LABEL_119:
      if ( *(char *)(v30 - 17) < 0 )
      {
        v44 = sub_BD2BC0(v39);
        v46 = v44 + v45;
        if ( *(char *)(v30 - 17) >= 0 )
        {
          if ( (unsigned int)(v46 >> 4) )
            goto LABEL_199;
        }
        else if ( (unsigned int)((v46 - sub_BD2BC0(v39)) >> 4) )
        {
          if ( *(char *)(v30 - 17) >= 0 )
            goto LABEL_199;
          v47 = *(_DWORD *)(sub_BD2BC0(v39) + 8);
          if ( *(char *)(v30 - 17) >= 0 )
            goto LABEL_197;
          v48 = sub_BD2BC0(v39);
          v50 = 32LL * (unsigned int)(*(_DWORD *)(v48 + v49 - 4) - v47);
          goto LABEL_125;
        }
      }
      v50 = 0;
LABEL_125:
      if ( (unsigned int)qword_502EE28 < (unsigned int)((32LL * (*(_DWORD *)(v30 - 20) & 0x7FFFFFF) - 32 - v76 - v50) >> 5) )
        a1[42] += a3;
      v51 = *(unsigned __int8 *)(v30 - 24);
      if ( v51 == 40 )
      {
        v52 = -32 - 32LL * (unsigned int)sub_B491D0(v39);
      }
      else
      {
        v52 = -32;
        if ( v51 != 85 )
        {
          if ( v51 != 34 )
            goto LABEL_197;
          v52 = -96;
        }
      }
      if ( *(char *)(v30 - 17) < 0 )
      {
        v53 = sub_BD2BC0(v39);
        v77 = v54 + v53;
        if ( *(char *)(v30 - 17) >= 0 )
        {
          if ( (unsigned int)(v77 >> 4) )
LABEL_199:
            BUG();
        }
        else if ( (unsigned int)((v77 - sub_BD2BC0(v39)) >> 4) )
        {
          if ( *(char *)(v30 - 17) >= 0 )
            goto LABEL_199;
          v73 = *(_DWORD *)(sub_BD2BC0(v39) + 8);
          if ( *(char *)(v30 - 17) >= 0 )
LABEL_197:
            BUG();
          v55 = sub_BD2BC0(v39);
          v52 -= 32LL * (unsigned int)(*(_DWORD *)(v55 + v56 - 4) - v73);
        }
      }
      v57 = v39 + v52;
      v32 = *(_DWORD *)(v30 - 20) & 0x7FFFFFF;
      v58 = v39 - 32 * v32;
      if ( v58 == v57 )
      {
LABEL_151:
        v29 = v80;
      }
      else
      {
        while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v58 + 8LL) + 8LL) != 14 )
        {
          v58 += 32;
          if ( v57 == v58 )
            goto LABEL_151;
        }
        a1[43] += a3;
        v29 = v80;
        v32 = *(_DWORD *)(v30 - 20) & 0x7FFFFFF;
      }
LABEL_66:
      v33 = v30 - 24;
      for ( j = 0; (unsigned int)v32 > (unsigned int)j; v32 = *(_DWORD *)(v30 - 20) & 0x7FFFFFF )
      {
        while ( 1 )
        {
          v35 = (*(_BYTE *)(v30 - 17) & 0x40) != 0 ? *(_QWORD *)(v30 - 32) : v33 - 32 * v32;
          v36 = **(_BYTE **)(v35 + 32LL * (_QWORD)j);
          if ( v36 > 3u )
            break;
          a1[26] += a3;
LABEL_71:
          ++j;
          v32 = *(_DWORD *)(v30 - 20) & 0x7FFFFFF;
          if ( (unsigned int)v32 <= (unsigned int)j )
            goto LABEL_84;
        }
        if ( v36 == 17 )
        {
          a1[21] += a3;
          goto LABEL_71;
        }
        if ( v36 == 18 )
        {
          a1[22] += a3;
          goto LABEL_71;
        }
        if ( v36 <= 0x15u )
        {
          a1[23] += a3;
          goto LABEL_71;
        }
        if ( v36 > 0x1Cu )
        {
          a1[24] += a3;
          goto LABEL_71;
        }
        switch ( v36 )
        {
          case 0x17u:
            a1[25] += a3;
            goto LABEL_71;
          case 0x19u:
            a1[27] += a3;
            goto LABEL_71;
          case 0x16u:
            a1[28] += a3;
            goto LABEL_71;
        }
        a1[29] += a3;
        ++j;
      }
LABEL_84:
      v29 = *(_QWORD *)(v29 + 8);
      v37 = 0;
      v80 = v29;
      v30 = v29;
      if ( v29 != v81 )
        break;
LABEL_90:
      if ( v85.m128i_i64[0] == v30 )
        goto LABEL_91;
    }
    while ( 1 )
    {
      if ( v30 )
        v30 -= 24LL;
      if ( !v83 )
        sub_4263D6(j, v30, v37);
      j = v82;
      if ( v84(v82, v30, v37, v26, v29, v33) )
        break;
      v30 = *(_QWORD *)(v80 + 8);
      v80 = v30;
      v29 = v30;
      if ( v81 == v30 )
        goto LABEL_90;
    }
  }
LABEL_91:
  if ( v88 )
    v88(v87, v87, 3);
  if ( v83 )
    v83(v82, v82, 3);
  if ( v98 )
    v98(v97, v97, 3);
  result = v93;
  if ( v93 )
    return (_QWORD *)((__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))v93)(v92, v92, 3);
  return result;
}
