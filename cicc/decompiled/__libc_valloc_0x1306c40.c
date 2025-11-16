// Function: __libc_valloc
// Address: 0x1306c40
//
// Alternative name is 'valloc'
void *__fastcall _libc_valloc(unsigned __int64 a1)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r13
  void *v5; // r14
  __int64 v6; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdi
  unsigned __int8 v10; // r15
  __int64 v11; // rcx
  size_t v12; // r13
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // rdx
  void *v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // [rsp+10h] [rbp-60h] BYREF
  __int128 v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]

  v2 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v9 = v2;
    v2 = sub_1313D30(v2, 0);
    if ( *(_BYTE *)(v2 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v9, 0) )
      {
        v5 = 0;
        *__errno_location() = 12;
        return v5;
      }
      v10 = unk_4F96994;
      if ( a1 <= 0x3800 )
      {
        v17 = (a1 + 4095) & 0xFFFFFFFFFFFFF000LL;
        if ( v17 > 0x1000 )
        {
          _BitScanReverse64(&v19, 2 * v17 - 1);
          v12 = -(1LL << ((unsigned __int8)v19 - 3)) & (v17 + (1LL << ((unsigned __int8)v19 - 3)) - 1);
        }
        else
        {
          v12 = qword_505FA40[byte_5060800[v17 >> 3]];
        }
        if ( v12 <= 0x3FFF )
          goto LABEL_28;
      }
      else if ( a1 > 0x4000 )
      {
        if ( a1 > 0x7000000000000000LL )
          goto LABEL_42;
        _BitScanReverse64((unsigned __int64 *)&v11, 2 * a1 - 1);
        if ( (unsigned __int64)(int)v11 < 7 )
          LOBYTE(v11) = 7;
        v12 = -(1LL << ((unsigned __int8)v11 - 3)) & ((1LL << ((unsigned __int8)v11 - 3)) + a1 - 1);
        if ( a1 > v12 || __CFADD__(unk_50607C0, v12) )
          goto LABEL_42;
LABEL_28:
        if ( v12 - 1 > 0x6FFFFFFFFFFFFFFFLL )
          goto LABEL_42;
LABEL_29:
        if ( *(char *)(v2 + 1) > 0 )
        {
          v13 = qword_50579C0[0];
          if ( qword_50579C0[0] || (v13 = sub_1300B80(v2, 0, (__int64)&off_49E8000)) != 0 )
          {
            v14 = 0;
            goto LABEL_31;
          }
          if ( !unk_505F9B8 )
            goto LABEL_42;
        }
        else
        {
          v13 = 0;
          v14 = v2 + 856;
          if ( *(_BYTE *)v2 )
            goto LABEL_31;
        }
        v14 = 0;
        v13 = 0;
LABEL_31:
        v5 = (void *)sub_1318040(v2, v13, v12, 4096, v10, v14);
        if ( v5 )
        {
          LOBYTE(v20) = 1;
          *(_QWORD *)&v21 = v2 + 824;
          *((_QWORD *)&v21 + 1) = v2 + 8;
          v22 = v2 + 16;
          v23 = v2 + 832;
          v15 = *(_QWORD *)(v2 + 824);
          *(_QWORD *)(v2 + 824) = v12 + v15;
          if ( v12 >= *(_QWORD *)(v2 + 16) - v15 )
            sub_13133F0(v2, &v20);
          v16 = v5;
          if ( !v10 && unk_4F969A2 )
          {
            off_4C6F0B8(v5, v12);
            v16 = v5;
          }
          goto LABEL_37;
        }
LABEL_42:
        v16 = 0;
        v5 = 0;
LABEL_37:
        v20 = a1;
        v21 = 0;
        sub_1346E80(5, v5, v16, &v20);
        return v5;
      }
      v12 = 0x4000;
      if ( (unsigned __int64)(unk_50607C0 + 0x4000LL) <= 0x3FFF )
        goto LABEL_42;
      goto LABEL_29;
    }
  }
  if ( a1 > 0x3800 )
  {
    if ( a1 <= 0x4000 )
      goto LABEL_6;
    if ( a1 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v8, 2 * a1 - 1);
    if ( (unsigned __int64)(int)v8 < 7 )
      LOBYTE(v8) = 7;
    v4 = -(1LL << ((unsigned __int8)v8 - 3)) & ((1LL << ((unsigned __int8)v8 - 3)) + a1 - 1);
    if ( a1 > v4 || __CFADD__(unk_50607C0, v4) )
      return 0;
LABEL_17:
    if ( v4 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    goto LABEL_7;
  }
  v3 = (a1 + 4095) & 0xFFFFFFFFFFFFF000LL;
  if ( v3 > 0x1000 )
  {
    _BitScanReverse64(&v18, 2 * v3 - 1);
    v4 = -(1LL << ((unsigned __int8)v18 - 3)) & (v3 + (1LL << ((unsigned __int8)v18 - 3)) - 1);
  }
  else
  {
    v4 = qword_505FA40[byte_5060800[v3 >> 3]];
  }
  if ( v4 <= 0x3FFF )
    goto LABEL_17;
LABEL_6:
  v4 = 0x4000;
  if ( (unsigned __int64)(unk_50607C0 + 0x4000LL) <= 0x3FFF )
    return 0;
LABEL_7:
  v5 = (void *)sub_1318040(v2, 0, v4, 4096, 0, v2 + 856);
  if ( !v5 )
    return 0;
  LOBYTE(v20) = 1;
  *(_QWORD *)&v21 = v2 + 824;
  *((_QWORD *)&v21 + 1) = v2 + 8;
  v22 = v2 + 16;
  v23 = v2 + 832;
  v6 = *(_QWORD *)(v2 + 824);
  *(_QWORD *)(v2 + 824) = v4 + v6;
  if ( v4 >= *(_QWORD *)(v2 + 16) - v6 )
    sub_13133F0(v2, &v20);
  return v5;
}
