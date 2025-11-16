// Function: sub_1318240
// Address: 0x1318240
//
void *__fastcall sub_1318240(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int8 a7,
        __int64 *a8,
        _BYTE *a9)
{
  unsigned __int64 v12; // r14
  unsigned __int8 v13; // r11
  char v14; // al
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  void *v17; // r13
  size_t v18; // rdx
  unsigned int v19; // r9d
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  unsigned int v28; // ecx
  __int64 v29; // r10
  __int64 *v30; // rbx
  void **v31; // rax
  void **v32; // rsi
  unsigned int v33; // eax
  char v34; // cl
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // r8d
  unsigned __int64 v40; // rcx
  _QWORD *v41; // r11
  unsigned __int64 v42; // rcx
  unsigned __int64 *v43; // rax
  unsigned __int64 v44; // rsi
  __int64 *v45; // r15
  __int64 v46; // rsi
  unsigned __int64 v47; // rcx
  char v48; // cl
  __int64 v49; // rax
  void **v50; // rax
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // rax
  _QWORD *v54; // r11
  unsigned __int64 v55; // rcx
  unsigned __int64 *v56; // rdx
  unsigned __int64 v57; // rdi
  __int64 *v58; // rax
  char v59; // cl
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  __int64 v62; // r13
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int64 v65; // rax
  __int64 i; // rdx
  int v67; // edi
  _QWORD *v68; // r8
  __int64 j; // rax
  int v70; // esi
  _QWORD *v71; // r8
  __int64 v72; // [rsp+8h] [rbp-1F8h]
  __int64 v73; // [rsp+10h] [rbp-1F0h]
  unsigned int v74; // [rsp+18h] [rbp-1E8h]
  __int64 v75; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v77; // [rsp+28h] [rbp-1D8h]
  unsigned int v78; // [rsp+28h] [rbp-1D8h]
  unsigned __int8 v79; // [rsp+34h] [rbp-1CCh]
  _QWORD v81[54]; // [rsp+50h] [rbp-1B0h] BYREF

  if ( a6 )
  {
    if ( a5 > 0x3800 || a6 > 0x1000 )
    {
      if ( a6 > 0x7000000000000000LL )
        return 0;
      if ( a5 > 0x4000 )
      {
        if ( a5 > 0x7000000000000000LL )
          return 0;
        _BitScanReverse64((unsigned __int64 *)&v25, 2 * a5 - 1);
        if ( (unsigned __int64)(int)v25 < 7 )
          LOBYTE(v25) = 7;
        v12 = -(1LL << ((unsigned __int8)v25 - 3)) & ((1LL << ((unsigned __int8)v25 - 3)) + a5 - 1);
        if ( a5 > v12 || __CFADD__(v12, ((a6 + 4095) & 0xFFFFFFFFFFFFF000LL) + *(_QWORD *)&dword_50607C0 - 4096) )
          return 0;
LABEL_38:
        v13 = a7;
        if ( a4 <= 0x3FFF )
          goto LABEL_49;
        return sub_130A210(a1, a2, a3, v12, a6, v13, (__int64)a8, a9);
      }
      if ( *(_QWORD *)&dword_50607C0 + ((a6 + 4095) & 0xFFFFFFFFFFFFF000LL) + 12288 <= 0x3FFF )
        return 0;
    }
    else
    {
      v23 = -(__int64)a6 & (a5 + a6 - 1);
      if ( v23 > 0x1000 )
      {
        _BitScanReverse64(&v40, 2 * v23 - 1);
        v12 = -(1LL << ((unsigned __int8)v40 - 3)) & (v23 + (1LL << ((unsigned __int8)v40 - 3)) - 1);
      }
      else
      {
        v12 = qword_505FA40[byte_5060800[(v23 + 7) >> 3]];
      }
      if ( v12 <= 0x3FFF )
        goto LABEL_4;
      if ( (unsigned __int64)(*(_QWORD *)&dword_50607C0 + 0x4000LL) <= 0x3FFF )
        return 0;
    }
    v12 = 0x4000;
    goto LABEL_38;
  }
  if ( a5 > 0x1000 )
  {
    if ( a5 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v26, 2 * a5 - 1);
    if ( (unsigned __int64)(int)v26 < 7 )
      LOBYTE(v26) = 7;
    v12 = -(1LL << ((unsigned __int8)v26 - 3)) & ((1LL << ((unsigned __int8)v26 - 3)) + a5 - 1);
  }
  else
  {
    v12 = qword_505FA40[byte_5060800[(a5 + 7) >> 3]];
  }
LABEL_4:
  if ( !v12 )
    return 0;
  v13 = a7;
  if ( v12 <= 0x3800 )
  {
    v14 = sub_1315EA0(a1, a3, a4, v12, 0, a7, v81);
    v13 = a7;
    if ( !v14 )
    {
      v17 = (void *)a3;
      sub_13470F0(*a9 ^ 1u, a3, a4, v12, a3, a9 + 8);
      return v17;
    }
    if ( a6 )
    {
      if ( a6 > 0x1000 )
      {
        if ( a6 > 0x7000000000000000LL )
          return 0;
LABEL_12:
        v16 = 0x4000;
        if ( ((a6 + 4095) & 0xFFFFFFFFFFFFF000LL) + *(_QWORD *)&dword_50607C0 + 12288 <= 0x3FFF )
          return 0;
        goto LABEL_13;
      }
      v15 = -(__int64)a6 & (v12 + a6 - 1);
      if ( v15 > 0x1000 )
      {
        _BitScanReverse64(&v47, 2 * v15 - 1);
        v16 = -(1LL << ((unsigned __int8)v47 - 3)) & (v15 + (1LL << ((unsigned __int8)v47 - 3)) - 1);
      }
      else
      {
        v16 = qword_505FA40[byte_5060800[(v15 + 7) >> 3]];
      }
      if ( v16 > 0x3FFF )
        goto LABEL_12;
LABEL_55:
      if ( v16 - 1 > 0x6FFFFFFFFFFFFFFFLL )
        return 0;
LABEL_13:
      v17 = (void *)sub_1318040(a1, a2, v16, a6, v13, a8);
      goto LABEL_14;
    }
    if ( v12 <= 0x1000 )
    {
      v28 = byte_5060800[(v12 + 7) >> 3];
      if ( a8 )
      {
LABEL_59:
        v29 = v28;
        v30 = &a8[3 * v28];
        v31 = (void **)v30[1];
        v17 = *v31;
        v32 = v31 + 1;
        if ( (_WORD)v31 != *((_WORD *)v30 + 12) )
        {
LABEL_60:
          v30[1] = (__int64)v32;
          goto LABEL_61;
        }
        if ( (_WORD)v31 == *((_WORD *)v30 + 14) )
        {
          v72 = v28;
          v73 = 3LL * v28;
          v74 = v28;
          v77 = v13;
          v51 = sub_1314520(a1, a2);
          v52 = v51;
          if ( !v51 )
            return 0;
          if ( !*(_WORD *)(unk_5060A20 + 2 * v72) )
          {
            v17 = (void *)sub_1317CF0(a1, v51, v12, v74, v77);
            goto LABEL_14;
          }
          sub_1310140(a1, a8, &a8[v73 + 1], v74, 1);
          v53 = sub_13100A0(a1, v52, a8, &a8[v73 + 1], v74, v81);
          v29 = v72;
          v17 = (void *)v53;
          if ( !LOBYTE(v81[0]) )
            return 0;
LABEL_61:
          if ( a7 )
            memset(v17, 0, qword_505FA40[v29]);
          ++v30[2];
LABEL_14:
          if ( v17 )
            goto LABEL_15;
          return 0;
        }
LABEL_105:
        v30[1] = (__int64)v32;
        *((_WORD *)v30 + 12) = (_WORD)v32;
        goto LABEL_61;
      }
LABEL_94:
      v17 = (void *)sub_1317CF0(a1, a2, v12, v28, v13);
      goto LABEL_14;
    }
    goto LABEL_96;
  }
  if ( a4 <= 0x3FFF || v12 <= 0x3FFF )
  {
    if ( !a6 )
    {
      if ( v12 > 0x7000000000000000LL )
      {
        v28 = 232;
        if ( !a8 )
          goto LABEL_94;
        goto LABEL_102;
      }
LABEL_96:
      v48 = 7;
      _BitScanReverse64((unsigned __int64 *)&v49, 2 * v12 - 1);
      if ( (unsigned int)v49 >= 7 )
        v48 = v49;
      if ( (unsigned int)v49 < 6 )
        LODWORD(v49) = 6;
      v28 = ((((v12 - 1) & (-1LL << (v48 - 3))) >> (v48 - 3)) & 3) + 4 * v49 - 23;
      if ( !a8 )
        goto LABEL_94;
      if ( v12 <= 0x3800 )
        goto LABEL_59;
LABEL_102:
      if ( v12 <= unk_5060A10 )
      {
        v29 = v28;
        v30 = &a8[3 * v28];
        v50 = (void **)v30[1];
        v17 = *v50;
        v32 = v50 + 1;
        if ( (_WORD)v50 != *((_WORD *)v30 + 12) )
          goto LABEL_60;
        if ( (_WORD)v50 == *((_WORD *)v30 + 14) )
        {
          v75 = 3LL * v28;
          v78 = v28;
          v79 = v13;
          v62 = sub_1314520(a1, a2);
          if ( v62 )
          {
            sub_1310140(a1, a8, &a8[v75 + 1], v78, 0);
            if ( v12 > 0x7000000000000000LL )
            {
              v64 = 0;
            }
            else
            {
              _BitScanReverse64((unsigned __int64 *)&v63, 2 * v12 - 1);
              if ( (unsigned __int64)(int)v63 < 7 )
                LOBYTE(v63) = 7;
              v64 = -(1LL << ((unsigned __int8)v63 - 3)) & (v12 + (1LL << ((unsigned __int8)v63 - 3)) - 1);
            }
            v17 = (void *)sub_1309DC0(a1, v62, v64, v79);
            if ( v17 )
            {
LABEL_15:
              sub_1346E80((unsigned int)(*a9 == 0) + 8, v17, v17, a9 + 8);
              sub_1346FC0((unsigned int)(*a9 == 0) + 3, a3, a9 + 8);
              v18 = v12;
              if ( a4 <= v12 )
                v18 = a4;
              memcpy(v17, (const void *)a3, v18);
              if ( a8 )
              {
                if ( a4 > 0x1000 )
                {
                  if ( a4 > 0x7000000000000000LL )
                  {
                    v19 = 232;
LABEL_78:
                    if ( v19 < dword_5060A18[0] )
                    {
                      v37 = 3LL * v19;
                      v21 = &a8[v37];
                      v22 = a8[v37 + 1];
                      if ( WORD1(a8[v37 + 3]) != (_WORD)v22 )
                        goto LABEL_22;
                      sub_1310E90(
                        a1,
                        a8,
                        (char **)&a8[v37 + 1],
                        v19,
                        (int)*(unsigned __int16 *)(unk_5060A20 + 2LL * v19) >> unk_4C6F1E8);
                      v38 = v21[1];
                      if ( *((_WORD *)v21 + 13) == (_WORD)v38 )
                        return v17;
LABEL_81:
                      v21[1] = v38 - 8;
                      *(_QWORD *)(v38 - 8) = a3;
                      return v17;
                    }
                    v41 = (_QWORD *)(a1 + 432);
                    if ( !a1 )
                    {
                      sub_130D500(v81);
                      v41 = v81;
                    }
                    v42 = a3 & 0xFFFFFFFFC0000000LL;
                    v43 = (_QWORD *)((char *)v41 + ((a3 >> 26) & 0xF0));
                    v44 = *v43;
                    if ( (a3 & 0xFFFFFFFFC0000000LL) == *v43 )
                    {
                      v45 = (__int64 *)(v43[1] + ((a3 >> 9) & 0x1FFFF8));
                    }
                    else if ( v42 == v41[32] )
                    {
                      v61 = v41[33];
LABEL_124:
                      v41[32] = v44;
                      v41[33] = v43[1];
                      v45 = (__int64 *)(v61 + ((a3 >> 9) & 0x1FFFF8));
                      *v43 = v42;
                      v43[1] = v61;
                    }
                    else
                    {
                      for ( i = 1; i != 8; ++i )
                      {
                        v67 = i;
                        if ( v42 == v41[2 * i + 32] )
                        {
                          v68 = &v41[2 * i];
                          v61 = v68[33];
                          v41 += 2 * (unsigned int)(v67 - 1);
                          v68[32] = v41[32];
                          v68[33] = v41[33];
                          goto LABEL_124;
                        }
                      }
                      v45 = (__int64 *)sub_130D370(a1, (__int64)&unk_5060AE0, v41, a3, 1, 0);
                    }
                    v46 = *v45;
LABEL_91:
                    sub_130A160(a1, (_QWORD *)((v46 << 16 >> 16) & 0xFFFFFFFFFFFFFF80LL));
                    return v17;
                  }
                  v34 = 7;
                  _BitScanReverse64((unsigned __int64 *)&v35, 2 * a4 - 1);
                  if ( (unsigned int)v35 >= 7 )
                    v34 = v35;
                  v36 = (((-1LL << (v34 - 3)) & (a4 - 1)) >> (v34 - 3)) & 3;
                  if ( (unsigned int)v35 < 6 )
                    LODWORD(v35) = 6;
                  v19 = v36 + 4 * v35 - 23;
                }
                else
                {
                  v19 = byte_5060800[(a4 + 7) >> 3];
                }
                if ( v19 <= 0x23 )
                {
                  v20 = 3LL * v19;
                  v21 = &a8[v20];
                  v22 = a8[v20 + 1];
                  if ( WORD1(a8[v20 + 3]) != (_WORD)v22 )
                  {
LABEL_22:
                    v21[1] = v22 - 8;
                    *(_QWORD *)(v22 - 8) = a3;
                    return v17;
                  }
                  v39 = *(unsigned __int16 *)(unk_5060A20 + 2LL * v19);
                  if ( (_WORD)v39 )
                  {
                    sub_13108D0(a1, a8, &a8[v20 + 1], v19, v39 >> unk_4C6F1EC);
                    v38 = v21[1];
                    if ( *((_WORD *)v21 + 13) == (_WORD)v38 )
                      return v17;
                    goto LABEL_81;
                  }
LABEL_70:
                  sub_1315B20(a1, a3);
                  return v17;
                }
                goto LABEL_78;
              }
              if ( a4 > 0x1000 )
              {
                if ( a4 > 0x7000000000000000LL )
                {
LABEL_111:
                  v54 = (_QWORD *)(a1 + 432);
                  if ( !a1 )
                  {
                    sub_130D500(v81);
                    v54 = v81;
                  }
                  v55 = a3 & 0xFFFFFFFFC0000000LL;
                  v56 = (_QWORD *)((char *)v54 + ((a3 >> 26) & 0xF0));
                  v57 = *v56;
                  if ( (a3 & 0xFFFFFFFFC0000000LL) == *v56 )
                  {
                    v58 = (__int64 *)(v56[1] + ((a3 >> 9) & 0x1FFFF8));
                  }
                  else if ( v55 == v54[32] )
                  {
                    v65 = v54[33];
LABEL_135:
                    v54[32] = v57;
                    v54[33] = v56[1];
                    v56[1] = v65;
                    *v56 = v55;
                    v58 = (__int64 *)(((a3 >> 9) & 0x1FFFF8) + v65);
                  }
                  else
                  {
                    for ( j = 1; j != 8; ++j )
                    {
                      v70 = j;
                      if ( v55 == v54[2 * j + 32] )
                      {
                        v71 = &v54[2 * j];
                        v65 = v71[33];
                        v54 += 2 * (unsigned int)(v70 - 1);
                        v71[32] = v54[32];
                        v71[33] = v54[33];
                        goto LABEL_135;
                      }
                    }
                    v58 = (__int64 *)sub_130D370(a1, (__int64)&unk_5060AE0, v54, a3, 1, 0);
                  }
                  v46 = *v58;
                  goto LABEL_91;
                }
                v59 = 7;
                _BitScanReverse64((unsigned __int64 *)&v60, 2 * a4 - 1);
                if ( (unsigned int)v60 >= 7 )
                  v59 = v60;
                if ( (unsigned int)v60 < 6 )
                  LODWORD(v60) = 6;
                v33 = ((((a4 - 1) & (-1LL << (v59 - 3))) >> (v59 - 3)) & 3) + 4 * v60 - 23;
              }
              else
              {
                v33 = byte_5060800[(a4 + 7) >> 3];
              }
              if ( v33 <= 0x23 )
                goto LABEL_70;
              goto LABEL_111;
            }
          }
          return 0;
        }
        goto LABEL_105;
      }
      goto LABEL_94;
    }
LABEL_49:
    if ( v12 <= 0x4000 )
      goto LABEL_12;
    if ( v12 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v27, 2 * v12 - 1);
    if ( (unsigned __int64)(int)v27 < 7 )
      LOBYTE(v27) = 7;
    v16 = (v12 + (1LL << ((unsigned __int8)v27 - 3)) - 1) & -(1LL << ((unsigned __int8)v27 - 3));
    if ( v12 > v16 || __CFADD__(v16, ((a6 + 4095) & 0xFFFFFFFFFFFFF000LL) + *(_QWORD *)&dword_50607C0 - 4096) )
      return 0;
    goto LABEL_55;
  }
  return sub_130A210(a1, a2, a3, v12, a6, v13, (__int64)a8, a9);
}
