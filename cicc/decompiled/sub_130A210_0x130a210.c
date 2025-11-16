// Function: sub_130A210
// Address: 0x130a210
//
void *__fastcall sub_130A210(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _BYTE *a8)
{
  __int64 v8; // r10
  unsigned __int8 v9; // r11
  _BYTE *v11; // r14
  unsigned __int64 v14; // rcx
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rax
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r14
  __int64 v20; // rax
  void *v21; // r15
  size_t v22; // rdx
  unsigned __int64 v23; // r12
  __int64 v24; // r10
  unsigned int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rdx
  unsigned __int64 v30; // rdx
  unsigned int v31; // eax
  _BYTE *v32; // r11
  unsigned __int64 v33; // rcx
  unsigned __int64 *v34; // rax
  unsigned __int64 v35; // rsi
  __int64 *v36; // r12
  __int64 v37; // rsi
  char v38; // cl
  __int64 v39; // rax
  unsigned __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rdx
  unsigned int i; // r8d
  _BYTE *v45; // rdi
  _BYTE *v46; // r11
  unsigned __int64 v47; // rdi
  unsigned __int64 *v48; // rax
  unsigned __int64 v49; // rcx
  __int64 *v50; // rax
  char v51; // cl
  __int64 v52; // rax
  unsigned __int64 v53; // r14
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rdx
  __int64 j; // rdx
  int v57; // edi
  _BYTE *v58; // r8
  __int64 k; // rdx
  int v60; // esi
  _BYTE *v61; // r8
  __int64 v62; // rax
  unsigned __int8 v64; // [rsp+10h] [rbp-1C0h]
  unsigned __int8 v66; // [rsp+10h] [rbp-1C0h]
  __int64 v67; // [rsp+18h] [rbp-1B8h]
  __int64 v68; // [rsp+18h] [rbp-1B8h]
  __int64 v69; // [rsp+18h] [rbp-1B8h]
  __int64 v70; // [rsp+18h] [rbp-1B8h]
  _BYTE v71[432]; // [rsp+20h] [rbp-1B0h] BYREF

  v8 = a1;
  v9 = a6;
  v11 = (_BYTE *)(a1 + 432);
  if ( !a1 )
  {
    v11 = v71;
    sub_130D500(v71);
    v9 = a6;
    v8 = 0;
  }
  v14 = a3 & 0xFFFFFFFFC0000000LL;
  v15 = (unsigned __int64 *)&v11[(a3 >> 26) & 0xF0];
  v16 = *v15;
  if ( (a3 & 0xFFFFFFFFC0000000LL) == *v15 )
  {
    v17 = (_QWORD *)(v15[1] + ((a3 >> 9) & 0x1FFFF8));
  }
  else if ( v14 == *((_QWORD *)v11 + 32) )
  {
    v30 = *((_QWORD *)v11 + 33);
LABEL_22:
    *((_QWORD *)v11 + 32) = v16;
    *((_QWORD *)v11 + 33) = v15[1];
    *v15 = v14;
    v15[1] = v30;
    v17 = (_QWORD *)(v30 + ((a3 >> 9) & 0x1FFFF8));
  }
  else
  {
    v43 = v11 + 272;
    for ( i = 1; i != 8; ++i )
    {
      if ( v14 == *v43 )
      {
        v45 = &v11[16 * i];
        v11 += 16 * i - 16;
        v30 = *((_QWORD *)v45 + 33);
        *((_QWORD *)v45 + 32) = *((_QWORD *)v11 + 32);
        *((_QWORD *)v45 + 33) = *((_QWORD *)v11 + 33);
        goto LABEL_22;
      }
      v43 += 2;
    }
    v66 = v9;
    v68 = v8;
    v17 = (_QWORD *)sub_130D370(v8, &unk_5060AE0, v11, a3, 1, 0);
    v9 = v66;
    v8 = v68;
  }
  v67 = v8;
  v64 = v9;
  v18 = ((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL;
  v19 = qword_505FA40[(unsigned __int8)(*(_QWORD *)v18 >> 20)];
  if ( !(unsigned __int8)sub_1309DD0(v8, (_QWORD *)v18, a4, a4, v9) )
  {
    sub_13470F0(*a8 ^ 1u, a3, v19, a4, a3, a8 + 8);
    return *(void **)(v18 + 8);
  }
  if ( a5 <= 0x40 )
    v20 = sub_1309DC0(v67, a2, a4, v64);
  else
    v20 = sub_1309830(v67, a2, a4, a5, v64);
  v21 = (void *)v20;
  if ( v20 )
  {
    sub_1346E80((unsigned int)(*a8 == 0) + 8, v20, v20, a8 + 8);
    sub_1346FC0((unsigned int)(*a8 == 0) + 3, a3, a8 + 8);
    v22 = a4;
    if ( a4 > v19 )
      v22 = v19;
    memcpy(v21, *(const void **)(v18 + 8), v22);
    v23 = *(_QWORD *)(v18 + 8);
    v24 = v67;
    if ( a7 )
    {
      if ( v19 > 0x1000 )
      {
        if ( v19 > 0x7000000000000000LL )
        {
          v25 = 232;
LABEL_29:
          if ( unk_5060A18 > v25 )
          {
            v41 = 24LL * v25;
            v27 = v41 + a7;
            v28 = *(_QWORD *)(v41 + a7 + 8);
            if ( *(_WORD *)(v41 + a7 + 26) != (_WORD)v28 )
              goto LABEL_16;
            sub_1310E90(v67, a7, a7 + v41 + 8, v25, (int)*(unsigned __int16 *)(unk_5060A20 + 2LL * v25) >> unk_4C6F1E8);
            v42 = *(_QWORD *)(v27 + 8);
            if ( *(_WORD *)(v27 + 26) == (_WORD)v42 )
              return v21;
LABEL_43:
            *(_QWORD *)(v27 + 8) = v42 - 8;
            *(_QWORD *)(v42 - 8) = v23;
            return v21;
          }
          v32 = (_BYTE *)(v67 + 432);
          if ( !v67 )
          {
            sub_130D500(v71);
            v24 = 0;
            v32 = v71;
          }
          v33 = v23 & 0xFFFFFFFFC0000000LL;
          v34 = (unsigned __int64 *)&v32[(v23 >> 26) & 0xF0];
          v35 = *v34;
          if ( (v23 & 0xFFFFFFFFC0000000LL) == *v34 )
          {
            v36 = (__int64 *)(v34[1] + ((v23 >> 9) & 0x1FFFF8));
          }
          else if ( v33 == *((_QWORD *)v32 + 32) )
          {
            v54 = *((_QWORD *)v32 + 33);
LABEL_63:
            *((_QWORD *)v32 + 32) = v35;
            *((_QWORD *)v32 + 33) = v34[1];
            v36 = (__int64 *)(v54 + ((v23 >> 9) & 0x1FFFF8));
            *v34 = v33;
            v34[1] = v54;
          }
          else
          {
            for ( j = 1; j != 8; ++j )
            {
              v57 = j;
              if ( v33 == *(_QWORD *)&v32[16 * j + 256] )
              {
                v58 = &v32[16 * j];
                v54 = *((_QWORD *)v58 + 33);
                v32 += 16 * (unsigned int)(v57 - 1);
                *((_QWORD *)v58 + 32) = *((_QWORD *)v32 + 32);
                *((_QWORD *)v58 + 33) = *((_QWORD *)v32 + 33);
                goto LABEL_63;
              }
            }
            v69 = v24;
            v62 = sub_130D370(v24, &unk_5060AE0, v32, v23, 1, 0);
            v24 = v69;
            v36 = (__int64 *)v62;
          }
          v37 = *v36;
LABEL_35:
          sub_130A160(v24, (_QWORD *)((v37 << 16 >> 16) & 0xFFFFFFFFFFFFFF80LL));
          return v21;
        }
        v38 = 7;
        _BitScanReverse64((unsigned __int64 *)&v39, 2 * v19 - 1);
        if ( (unsigned int)v39 >= 7 )
          v38 = v39;
        v40 = (((-1LL << (v38 - 3)) & (v19 - 1)) >> (v38 - 3)) & 3;
        if ( (unsigned int)v39 < 6 )
          LODWORD(v39) = 6;
        v25 = v40 + 4 * v39 - 23;
      }
      else
      {
        v25 = byte_5060800[(v19 + 7) >> 3];
      }
      if ( v25 <= 0x23 )
      {
        v26 = 24LL * v25;
        v27 = v26 + a7;
        v28 = *(_QWORD *)(v26 + a7 + 8);
        if ( *(_WORD *)(v26 + a7 + 26) != (_WORD)v28 )
        {
LABEL_16:
          *(_QWORD *)(v27 + 8) = v28 - 8;
          *(_QWORD *)(v28 - 8) = v23;
          return v21;
        }
        if ( *(_WORD *)(unk_5060A20 + 2LL * v25) )
        {
          sub_13108D0(v67, a7, a7 + v26 + 8, v25, (int)*(unsigned __int16 *)(unk_5060A20 + 2LL * v25) >> unk_4C6F1EC);
          v42 = *(_QWORD *)(v27 + 8);
          if ( *(_WORD *)(v27 + 26) == (_WORD)v42 )
            return v21;
          goto LABEL_43;
        }
LABEL_26:
        sub_1315B20(v67, v23);
        return v21;
      }
      goto LABEL_29;
    }
    if ( v19 > 0x1000 )
    {
      if ( v19 > 0x7000000000000000LL )
      {
LABEL_50:
        v46 = (_BYTE *)(v67 + 432);
        if ( !v67 )
        {
          sub_130D500(v71);
          v24 = 0;
          v46 = v71;
        }
        v47 = v23 & 0xFFFFFFFFC0000000LL;
        v48 = (unsigned __int64 *)&v46[(v23 >> 26) & 0xF0];
        v49 = *v48;
        if ( (v23 & 0xFFFFFFFFC0000000LL) == *v48 )
        {
          v50 = (__int64 *)(v48[1] + ((v23 >> 9) & 0x1FFFF8));
        }
        else if ( v47 == *((_QWORD *)v46 + 32) )
        {
          v55 = *((_QWORD *)v46 + 33);
LABEL_68:
          *((_QWORD *)v46 + 32) = v49;
          *((_QWORD *)v46 + 33) = v48[1];
          *v48 = v47;
          v48[1] = v55;
          v50 = (__int64 *)(v55 + ((v23 >> 9) & 0x1FFFF8));
        }
        else
        {
          for ( k = 1; k != 8; ++k )
          {
            v60 = k;
            if ( v47 == *(_QWORD *)&v46[16 * k + 256] )
            {
              v61 = &v46[16 * k];
              v55 = *((_QWORD *)v61 + 33);
              v46 += 16 * (unsigned int)(v60 - 1);
              *((_QWORD *)v61 + 32) = *((_QWORD *)v46 + 32);
              *((_QWORD *)v61 + 33) = *((_QWORD *)v46 + 33);
              goto LABEL_68;
            }
          }
          v70 = v24;
          v50 = (__int64 *)sub_130D370(v24, &unk_5060AE0, v46, v23, 1, 0);
          v24 = v70;
        }
        v37 = *v50;
        goto LABEL_35;
      }
      v51 = 7;
      _BitScanReverse64((unsigned __int64 *)&v52, 2 * v19 - 1);
      if ( (unsigned int)v52 >= 7 )
        v51 = v52;
      v53 = (((-1LL << (v51 - 3)) & (v19 - 1)) >> (v51 - 3)) & 3;
      if ( (unsigned int)v52 < 6 )
        LODWORD(v52) = 6;
      v31 = v53 + 4 * v52 - 23;
    }
    else
    {
      v31 = byte_5060800[(v19 + 7) >> 3];
    }
    if ( v31 <= 0x23 )
      goto LABEL_26;
    goto LABEL_50;
  }
  return v21;
}
