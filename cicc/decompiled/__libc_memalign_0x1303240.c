// Function: __libc_memalign
// Address: 0x1303240
//
// Alternative name is 'memalign'
void *__fastcall _libc_memalign(unsigned __int64 a1, unsigned __int64 a2)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r15
  void *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  size_t v13; // r15
  unsigned __int8 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // r9
  unsigned __int8 v17; // r10
  __int64 v18; // rdx
  void *v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  unsigned __int8 v24; // [rsp+8h] [rbp-68h]
  unsigned __int8 v25; // [rsp+8h] [rbp-68h]
  unsigned __int64 v26; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]

  v4 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v11 = v4;
    v4 = sub_1313D30(v4, 0);
    if ( *(_BYTE *)(v4 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v11, 0) )
      {
        v7 = 0;
        *__errno_location() = 12;
        return v7;
      }
      if ( !a1 || ((a1 - 1) & a1) != 0 )
        goto LABEL_50;
      if ( a2 > 0x3800 || a1 > 0x1000 )
      {
        if ( a1 > 0x7000000000000000LL )
          goto LABEL_50;
        if ( a2 <= 0x4000 )
        {
LABEL_32:
          v13 = 0x4000;
          if ( ((a1 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
            goto LABEL_50;
LABEL_33:
          v14 = unk_4F96994;
          if ( *(char *)(v4 + 1) > 0 )
          {
            v15 = qword_50579C0[0];
            if ( qword_50579C0[0]
              || (v25 = unk_4F96994, v22 = sub_1300B80(v4, 0, (__int64)&off_49E8000), v14 = v25, (v15 = v22) != 0) )
            {
              v16 = 0;
              goto LABEL_35;
            }
            if ( !unk_505F9B8 )
              goto LABEL_50;
          }
          else
          {
            v15 = 0;
            v16 = v4 + 856;
            if ( *(_BYTE *)v4 )
              goto LABEL_35;
          }
          v16 = 0;
          v15 = 0;
LABEL_35:
          v24 = v14;
          v7 = (void *)sub_1318040(v4, v15, v13, a1, v14, v16);
          if ( v7 )
          {
            LOBYTE(v26) = 1;
            v17 = v24;
            v27 = v4 + 824;
            v28 = v4 + 8;
            v29 = v4 + 16;
            v30 = v4 + 832;
            v18 = *(_QWORD *)(v4 + 824);
            *(_QWORD *)(v4 + 824) = v13 + v18;
            if ( v13 >= *(_QWORD *)(v4 + 16) - v18 )
            {
              sub_13133F0(v4, &v26);
              v17 = v24;
            }
            v19 = v7;
            if ( !v17 && unk_4F969A2 )
            {
              off_4C6F0B8(v7, v13);
              v19 = v7;
            }
            goto LABEL_41;
          }
LABEL_50:
          v19 = 0;
          v7 = 0;
LABEL_41:
          v28 = 0;
          v26 = a1;
          v27 = a2;
          sub_1346E80(4, v7, v19, &v26);
          return v7;
        }
        if ( a2 > 0x7000000000000000LL )
          goto LABEL_50;
        _BitScanReverse64((unsigned __int64 *)&v20, 2 * a2 - 1);
        if ( (unsigned __int64)(int)v20 < 7 )
          LOBYTE(v20) = 7;
        v13 = -(1LL << ((unsigned __int8)v20 - 3)) & ((1LL << ((unsigned __int8)v20 - 3)) + a2 - 1);
        if ( a2 > v13 || __CFADD__(v13, unk_50607C0 + ((a1 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
          goto LABEL_50;
      }
      else
      {
        v12 = -(__int64)a1 & (a2 + a1 - 1);
        if ( v12 > 0x1000 )
        {
          _BitScanReverse64(&v23, 2 * v12 - 1);
          v13 = -(1LL << ((unsigned __int8)v23 - 3)) & (v12 + (1LL << ((unsigned __int8)v23 - 3)) - 1);
        }
        else
        {
          v13 = qword_505FA40[byte_5060800[(v12 + 7) >> 3]];
        }
        if ( v13 > 0x3FFF )
          goto LABEL_32;
      }
      if ( v13 - 1 > 0x6FFFFFFFFFFFFFFFLL )
        goto LABEL_50;
      goto LABEL_33;
    }
  }
  if ( !a1 || ((a1 - 1) & a1) != 0 )
    return 0;
  if ( a2 <= 0x3800 && a1 <= 0x1000 )
  {
    v5 = -(__int64)a1 & (a2 + a1 - 1);
    if ( v5 > 0x1000 )
    {
      _BitScanReverse64(&v21, 2 * v5 - 1);
      v6 = -(1LL << ((unsigned __int8)v21 - 3)) & (v5 + (1LL << ((unsigned __int8)v21 - 3)) - 1);
    }
    else
    {
      v6 = qword_505FA40[byte_5060800[(v5 + 7) >> 3]];
    }
    if ( v6 > 0x3FFF )
      goto LABEL_9;
LABEL_20:
    if ( v6 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    goto LABEL_10;
  }
  if ( a1 > 0x7000000000000000LL )
    return 0;
  if ( a2 > 0x4000 )
  {
    if ( a2 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v9, 2 * a2 - 1);
    if ( (unsigned __int64)(int)v9 < 7 )
      LOBYTE(v9) = 7;
    v6 = -(1LL << ((unsigned __int8)v9 - 3)) & ((1LL << ((unsigned __int8)v9 - 3)) + a2 - 1);
    if ( a2 > v6 || __CFADD__(v6, ((a1 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 - 4096) )
      return 0;
    goto LABEL_20;
  }
LABEL_9:
  v6 = 0x4000;
  if ( unk_50607C0 + ((a1 + 4095) & 0xFFFFFFFFFFFFF000LL) + 12288 <= 0x3FFF )
    return 0;
LABEL_10:
  v7 = (void *)sub_1318040(v4, 0, v6, a1, 0, v4 + 856);
  if ( !v7 )
    return 0;
  LOBYTE(v26) = 1;
  v27 = v4 + 824;
  v28 = v4 + 8;
  v29 = v4 + 16;
  v30 = v4 + 832;
  v8 = *(_QWORD *)(v4 + 824);
  *(_QWORD *)(v4 + 824) = v6 + v8;
  if ( v6 >= *(_QWORD *)(v4 + 16) - v8 )
    sub_13133F0(v4, &v26);
  return v7;
}
