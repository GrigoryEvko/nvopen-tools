// Function: sub_13010C0
// Address: 0x13010c0
//
char *__fastcall sub_13010C0(unsigned __int64 src, unsigned __int64 a2, signed int a3, char a4, int a5, int a6)
{
  __int64 v8; // rbx
  unsigned int v9; // ecx
  unsigned __int64 v10; // r13
  __int64 v11; // r12
  unsigned __int8 v12; // dl
  __int64 v13; // rax
  char *v14; // r10
  unsigned __int64 *v15; // rcx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  size_t v18; // r11
  unsigned __int64 v19; // r10
  __int64 v20; // r8
  __int64 v21; // rax
  size_t v22; // r11
  size_t v23; // r10
  char *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx
  char *v32; // rax
  size_t v33; // rdx
  unsigned int v34; // r9d
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned int v39; // esi
  __int64 *v40; // rdx
  unsigned __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rax
  char v47; // cl
  __int64 v48; // rax
  unsigned __int16 v49; // dx
  unsigned __int64 v50; // rcx
  _QWORD *v51; // rdx
  unsigned int i; // esi
  char *v53; // rdi
  char *v54; // r10
  char *v55; // rdx
  char *v56; // rcx
  __int64 v57; // rsi
  _QWORD *v58; // rax
  unsigned __int64 v59; // rbx
  __int64 v60; // rax
  char v61; // cl
  __int64 v62; // rax
  __int64 v63; // rax
  int v64; // edi
  __int64 v65; // r8
  char *v66; // r8
  __int64 v67; // rax
  __int64 v68; // [rsp-18h] [rbp-238h]
  size_t n; // [rsp+8h] [rbp-218h]
  unsigned __int64 v70; // [rsp+10h] [rbp-210h]
  unsigned __int64 v71; // [rsp+18h] [rbp-208h]
  __int64 *v72; // [rsp+18h] [rbp-208h]
  __int64 v73; // [rsp+20h] [rbp-200h]
  unsigned __int64 v75; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v76; // [rsp+28h] [rbp-1F8h]
  size_t v77; // [rsp+28h] [rbp-1F8h]
  size_t v78; // [rsp+28h] [rbp-1F8h]
  size_t v79; // [rsp+28h] [rbp-1F8h]
  size_t v80; // [rsp+28h] [rbp-1F8h]
  __int64 v81; // [rsp+30h] [rbp-1F0h]
  size_t v82; // [rsp+30h] [rbp-1F0h]
  size_t v83; // [rsp+30h] [rbp-1F0h]
  size_t v84; // [rsp+30h] [rbp-1F0h]
  size_t v85; // [rsp+30h] [rbp-1F0h]
  size_t v86; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 v87; // [rsp+3Fh] [rbp-1E1h]
  char v88[8]; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int64 v89; // [rsp+48h] [rbp-1D8h] BYREF
  unsigned __int64 v90; // [rsp+50h] [rbp-1D0h]
  __int64 v91; // [rsp+58h] [rbp-1C8h]
  __int64 v92; // [rsp+60h] [rbp-1C0h]
  char v93[8]; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v94; // [rsp+78h] [rbp-1A8h]
  __int64 v95; // [rsp+80h] [rbp-1A0h]
  __int64 v96; // [rsp+88h] [rbp-198h]
  __int64 v97; // [rsp+90h] [rbp-190h]

  v8 = a3;
  v9 = a3;
  v10 = (1LL << a3) & 0xFFFFFFFFFFFFFFFELL;
  v11 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v11 = sub_1313D30(v11, 0);
  v12 = unk_4F96994;
  if ( !unk_4F96994 )
    v12 = (v8 & 0x40) != 0;
  v87 = v12;
  if ( (v8 & 0xFFF00000) == 0 )
  {
LABEL_6:
    v73 = 0;
    goto LABEL_7;
  }
  v9 = ((unsigned int)v8 >> 20) - 1;
  v73 = qword_50579C0[v9];
  if ( !v73 )
  {
    v73 = sub_1300B80(v11, v9, (__int64)&off_49E8000);
    if ( !v73 )
    {
      v9 = ((unsigned int)v8 >> 20) - 1;
      if ( unk_505F9B8 <= v9 )
        return 0;
      goto LABEL_6;
    }
  }
LABEL_7:
  if ( (v8 & 0xFFF00) == 0 )
  {
LABEL_8:
    v13 = 0;
    if ( *(_BYTE *)v11 )
      v13 = v11 + 856;
    v81 = v13;
    goto LABEL_11;
  }
  if ( (v8 & 0xFFF00) == 0x100 )
    goto LABEL_77;
  v39 = (((int)v8 >> 8) & 0xFFF) - 2;
  if ( (((int)v8 >> 8) & 0xFFF) == 0 )
    goto LABEL_8;
  if ( (((int)v8 >> 8) & 0xFFF) == 1 )
  {
LABEL_77:
    v81 = 0;
  }
  else
  {
    v40 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v39);
    v81 = *v40;
    if ( !*v40 )
    {
      sub_130ACF0((unsigned int)"<jemalloc>: invalid tcache id (%u).\n", v39, (_DWORD)v40, v9, a5, a6);
      abort();
    }
    if ( v81 == 1 )
    {
      v72 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v39);
      v81 = sub_1311F90(v11);
      *v72 = v81;
    }
  }
  if ( v11 )
  {
LABEL_11:
    v14 = (char *)(v11 + 432);
    goto LABEL_12;
  }
  sub_130D500(v93);
  v14 = v93;
LABEL_12:
  v71 = src & 0xFFFFFFFFC0000000LL;
  v15 = (unsigned __int64 *)&v14[(src >> 26) & 0xF0];
  v70 = (src >> 26) & 0xF0;
  v16 = *v15;
  if ( (src & 0xFFFFFFFFC0000000LL) == *v15 )
  {
    v17 = (_QWORD *)(v15[1] + ((src >> 9) & 0x1FFFF8));
  }
  else if ( v71 == *((_QWORD *)v14 + 32) )
  {
    *((_QWORD *)v14 + 32) = v16;
    v41 = *((_QWORD *)v14 + 33);
    *((_QWORD *)v14 + 33) = v15[1];
    *v15 = v71;
LABEL_76:
    v15[1] = v41;
    v17 = (_QWORD *)(v41 + ((src >> 9) & 0x1FFFF8));
  }
  else
  {
    v51 = v14 + 272;
    for ( i = 1; i != 8; ++i )
    {
      if ( v71 == *v51 )
      {
        v53 = &v14[16 * i];
        v54 = &v14[16 * i - 16];
        v41 = *((_QWORD *)v53 + 33);
        *((_QWORD *)v53 + 32) = *((_QWORD *)v54 + 32);
        *((_QWORD *)v53 + 33) = *((_QWORD *)v54 + 33);
        *((_QWORD *)v54 + 32) = v16;
        *((_QWORD *)v54 + 33) = v15[1];
        *v15 = v71;
        goto LABEL_76;
      }
      v51 += 2;
    }
    v17 = (_QWORD *)sub_130D370(v11, &unk_5060AE0, v14, src, 1, 0);
  }
  v18 = qword_505FA40[HIWORD(*v17)];
  if ( v10 )
  {
    if ( v10 <= 0x1000 && a2 <= 0x3800 )
    {
      v28 = -(__int64)v10 & (v10 + a2 - 1);
      if ( v28 > 0x1000 )
      {
        _BitScanReverse64(&v43, 2 * v28 - 1);
        v19 = -(1LL << ((unsigned __int8)v43 - 3)) & (v28 + (1LL << ((unsigned __int8)v43 - 3)) - 1);
      }
      else
      {
        v19 = qword_505FA40[byte_5060800[(v28 + 7) >> 3]];
      }
      if ( v19 > 0x3FFF )
        goto LABEL_32;
    }
    else
    {
      if ( v10 > 0x7000000000000000LL )
        return 0;
      if ( a2 <= 0x4000 )
      {
LABEL_32:
        v19 = 0x4000;
        if ( unk_50607C0 + ((v10 + 4095) & 0xFFFFFFFFFFFFF000LL) + 12288 <= 0x3FFF )
          return 0;
LABEL_42:
        v89 = src;
        v90 = a2;
        v20 = v87;
        v88[0] = a4;
        v91 = v8;
        v92 = 0;
        if ( ((v10 - 1) & src) == 0 )
          goto LABEL_19;
        if ( v10 <= 0x1000 && a2 <= 0x3800 )
        {
          v30 = -(__int64)v10 & (v10 + a2 - 1);
          if ( v30 > 0x1000 )
          {
            _BitScanReverse64(&v50, 2 * v30 - 1);
            v31 = -(1LL << ((unsigned __int8)v50 - 3)) & (v30 + (1LL << ((unsigned __int8)v50 - 3)) - 1);
          }
          else
          {
            v31 = qword_505FA40[byte_5060800[(v30 + 7) >> 3]];
          }
          if ( v31 > 0x3FFF )
            goto LABEL_47;
        }
        else
        {
          if ( v10 > 0x7000000000000000LL )
            return 0;
          if ( a2 <= 0x4000 )
          {
LABEL_47:
            v31 = 0x4000;
            if ( ((v10 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
              return 0;
LABEL_48:
            n = v18;
            v76 = v19;
            v32 = (char *)sub_1318040(v11, v73, v31, v10, v87, v81);
            v24 = v32;
            if ( !v32 )
              return 0;
            v33 = a2;
            if ( a2 > n )
              v33 = n;
            memcpy(v32, (const void *)src, v33);
            sub_1346E80((unsigned int)(v88[0] == 0) + 8, v24, v24, &v89);
            sub_1346FC0((unsigned int)(v88[0] == 0) + 3, src, &v89);
            v22 = n;
            v23 = v76;
            if ( v81 )
            {
              if ( n > 0x1000 )
              {
                if ( n > 0x7000000000000000LL )
                {
                  v34 = 232;
                  goto LABEL_92;
                }
                v47 = 7;
                _BitScanReverse64((unsigned __int64 *)&v48, 2 * n - 1);
                if ( (unsigned int)v48 >= 7 )
                  v47 = v48;
                if ( (unsigned int)v48 < 6 )
                  LODWORD(v48) = 6;
                v34 = ((((n - 1) & (-1LL << (v47 - 3))) >> (v47 - 3)) & 3) + 4 * v48 - 23;
              }
              else
              {
                v34 = byte_5060800[(n + 7) >> 3];
              }
              if ( v34 <= 0x23 )
              {
                v35 = 24LL * v34;
                v36 = v35 + v81;
                v37 = *(_QWORD *)(v35 + v81 + 8);
                if ( *(_WORD *)(v35 + v81 + 26) != (_WORD)v37 )
                {
LABEL_56:
                  *(_QWORD *)(v36 + 8) = v37 - 8;
                  *(_QWORD *)(v37 - 8) = src;
                  goto LABEL_20;
                }
                v49 = *(_WORD *)(unk_5060A20 + 2LL * v34);
                if ( v49 )
                {
                  sub_13108D0(v11, v81, v81 + v35 + 8, v34, (int)v49 >> unk_4C6F1EC);
                  goto LABEL_95;
                }
LABEL_90:
                sub_1315B20(v11, src);
                v23 = v76;
                v22 = n;
                goto LABEL_20;
              }
LABEL_92:
              if ( v34 < unk_5060A18 )
              {
                v45 = 24LL * v34;
                v36 = v45 + v81;
                v37 = *(_QWORD *)(v45 + v81 + 8);
                if ( *(_WORD *)(v45 + v81 + 26) != (_WORD)v37 )
                  goto LABEL_56;
                sub_1310E90(
                  v11,
                  v81,
                  v81 + v45 + 8,
                  v34,
                  (int)*(unsigned __int16 *)(unk_5060A20 + 2LL * v34) >> unk_4C6F1E8);
LABEL_95:
                v46 = *(_QWORD *)(v36 + 8);
                v23 = v76;
                v22 = n;
                if ( *(_WORD *)(v36 + 26) != (_WORD)v46 )
                {
                  *(_QWORD *)(v36 + 8) = v46 - 8;
                  *(_QWORD *)(v46 - 8) = src;
                }
                goto LABEL_20;
              }
              v55 = (char *)(v11 + 432);
              if ( !v11 )
              {
                sub_130D500(v93);
                v22 = n;
                v23 = v76;
                v55 = v93;
              }
              v56 = &v55[v70];
              v57 = *(_QWORD *)&v55[v70];
              if ( v71 == v57 )
                goto LABEL_112;
              v59 = v71;
              if ( v71 != *((_QWORD *)v55 + 32) )
              {
                v63 = 1;
                while ( 1 )
                {
                  v59 = v71;
                  v64 = v63;
                  v65 = 16 * v63;
                  if ( v71 == *(_QWORD *)&v55[16 * v63 + 256] )
                    break;
                  if ( ++v63 == 8 )
                    goto LABEL_134;
                }
LABEL_132:
                v66 = &v55[v65];
                v60 = *((_QWORD *)v66 + 33);
                v55 += 16 * (unsigned int)(v64 - 1);
                *((_QWORD *)v66 + 32) = *((_QWORD *)v55 + 32);
                *((_QWORD *)v66 + 33) = *((_QWORD *)v55 + 33);
                goto LABEL_120;
              }
              goto LABEL_119;
            }
            if ( n > 0x1000 )
            {
              if ( n > 0x7000000000000000LL )
                goto LABEL_115;
              v61 = 7;
              _BitScanReverse64((unsigned __int64 *)&v62, 2 * n - 1);
              if ( (unsigned int)v62 >= 7 )
                v61 = v62;
              if ( (unsigned int)v62 < 6 )
                LODWORD(v62) = 6;
              v44 = ((((n - 1) & (-1LL << (v61 - 3))) >> (v61 - 3)) & 3) + 4 * v62 - 23;
            }
            else
            {
              v44 = byte_5060800[(n + 7) >> 3];
            }
            if ( v44 <= 0x23 )
              goto LABEL_90;
LABEL_115:
            v55 = (char *)(v11 + 432);
            if ( !v11 )
            {
              sub_130D500(v93);
              v22 = n;
              v23 = v76;
              v55 = v93;
            }
            v56 = &v55[v70];
            v57 = *(_QWORD *)&v55[v70];
            if ( v71 == v57 )
            {
LABEL_112:
              v58 = (_QWORD *)(*((_QWORD *)v56 + 1) + ((src >> 9) & 0x1FFFF8));
LABEL_113:
              v79 = v22;
              v85 = v23;
              sub_130A160(v11, ((__int64)(*v58 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
              v23 = v85;
              v22 = v79;
              goto LABEL_20;
            }
            v59 = v71;
            if ( v71 != *((_QWORD *)v55 + 32) )
            {
              v67 = 1;
              while ( 1 )
              {
                v59 = v71;
                v64 = v67;
                v65 = 16 * v67;
                if ( v71 == *(_QWORD *)&v55[16 * v67 + 256] )
                  goto LABEL_132;
                if ( ++v67 == 8 )
                {
LABEL_134:
                  v80 = v23;
                  v86 = v22;
                  v58 = (_QWORD *)sub_130D370(v11, &unk_5060AE0, v55, src, 1, 0);
                  v23 = v80;
                  v22 = v86;
                  goto LABEL_113;
                }
              }
            }
LABEL_119:
            v60 = *((_QWORD *)v55 + 33);
LABEL_120:
            *((_QWORD *)v55 + 32) = v57;
            *((_QWORD *)v55 + 33) = *((_QWORD *)v56 + 1);
            *((_QWORD *)v56 + 1) = v60;
            *(_QWORD *)v56 = v59;
            v58 = (_QWORD *)(((src >> 9) & 0x1FFFF8) + v60);
            goto LABEL_113;
          }
          if ( a2 > 0x7000000000000000LL )
            return 0;
          _BitScanReverse64((unsigned __int64 *)&v38, 2 * a2 - 1);
          if ( (unsigned __int64)(int)v38 < 7 )
            LOBYTE(v38) = 7;
          v31 = ((1LL << ((unsigned __int8)v38 - 3)) + a2 - 1) & -(1LL << ((unsigned __int8)v38 - 3));
          if ( a2 > v31 || __CFADD__(v31, unk_50607C0 + ((v10 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
            return 0;
        }
        if ( v31 - 1 > 0x6FFFFFFFFFFFFFFFLL )
          return 0;
        goto LABEL_48;
      }
      if ( a2 > 0x7000000000000000LL )
        return 0;
      _BitScanReverse64((unsigned __int64 *)&v29, 2 * a2 - 1);
      if ( (unsigned __int64)(int)v29 < 7 )
        LOBYTE(v29) = 7;
      v19 = -(1LL << ((unsigned __int8)v29 - 3)) & ((1LL << ((unsigned __int8)v29 - 3)) + a2 - 1);
      if ( a2 > v19 || __CFADD__(v19, unk_50607C0 + ((v10 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
        return 0;
    }
    if ( v19 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    goto LABEL_42;
  }
  if ( a2 > 0x1000 )
  {
    if ( a2 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v42, 2 * a2 - 1);
    if ( (unsigned __int64)(int)v42 < 7 )
      LOBYTE(v42) = 7;
    v19 = -(1LL << ((unsigned __int8)v42 - 3)) & ((1LL << ((unsigned __int8)v42 - 3)) + a2 - 1);
  }
  else
  {
    v19 = qword_505FA40[byte_5060800[(a2 + 7) >> 3]];
  }
  if ( v19 - 1 > 0x6FFFFFFFFFFFFFFFLL )
    return 0;
  v89 = src;
  v90 = a2;
  v20 = v87;
  v88[0] = a4;
  v91 = v8;
  v92 = 0;
LABEL_19:
  v75 = v19;
  v68 = v81;
  v82 = v18;
  v21 = sub_1318240(v11, v73, src, v18, a2, v10, v20, v68, v88);
  v22 = v82;
  v23 = v75;
  v24 = (char *)v21;
  if ( !v21 )
    return 0;
LABEL_20:
  v93[0] = 1;
  v94 = v11 + 824;
  v95 = v11 + 8;
  v96 = v11 + 16;
  v97 = v11 + 832;
  v25 = *(_QWORD *)(v11 + 824);
  *(_QWORD *)(v11 + 824) = v25 + v23;
  if ( *(_QWORD *)(v11 + 16) - v25 <= v23 )
  {
    v78 = v22;
    v84 = v23;
    sub_13133F0(v11, v93);
    v22 = v78;
    v23 = v84;
  }
  v93[0] = 0;
  v94 = v11 + 840;
  v95 = v11 + 24;
  v96 = v11 + 32;
  v97 = v11 + 848;
  v26 = *(_QWORD *)(v11 + 840);
  *(_QWORD *)(v11 + 840) = v22 + v26;
  if ( v22 >= *(_QWORD *)(v11 + 32) - v26 )
  {
    v77 = v22;
    v83 = v23;
    sub_13133F0(v11, v93);
    v22 = v77;
    v23 = v83;
  }
  if ( (unk_4F969A2 & (v22 < v23)) != 0 && !v87 )
    off_4C6F0B8(&v24[v22], v23 - v22);
  return v24;
}
