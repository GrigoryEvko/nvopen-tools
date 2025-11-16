// Function: pvalloc
// Address: 0x1307120
//
// Alternative name is '__libc_pvalloc'
void *__fastcall pvalloc(__int64 a1)
{
  unsigned __int64 v1; // rbx
  __int64 v2; // r12
  unsigned __int64 v3; // r13
  void *v4; // r14
  __int64 v5; // rdx
  __int64 v7; // rcx
  __int64 v9; // rdi
  unsigned __int8 v10; // r10
  __int64 v11; // rcx
  size_t v12; // r15
  __int64 v13; // rsi
  __int64 v14; // r9
  unsigned __int8 v15; // r10
  __int64 v16; // rdx
  void *v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  unsigned __int8 v21; // [rsp+8h] [rbp-68h]
  unsigned __int8 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  __int128 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]

  v1 = (a1 + 4095) & 0xFFFFFFFFFFFFF000LL;
  v2 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v9 = v2;
    v2 = sub_1313D30(v2, 0);
    if ( *(_BYTE *)(v2 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v9, 0) )
      {
        v4 = 0;
        *__errno_location() = 12;
        return v4;
      }
      v10 = unk_4F96994;
      if ( v1 <= 0x3800 )
      {
        if ( v1 > 0x1000 )
        {
          _BitScanReverse64(&v20, 2 * v1 - 1);
          v12 = -(1LL << ((unsigned __int8)v20 - 3)) & (v1 + (1LL << ((unsigned __int8)v20 - 3)) - 1);
        }
        else
        {
          v12 = qword_505FA40[byte_5060800[v1 >> 3]];
        }
        if ( v12 <= 0x3FFF )
          goto LABEL_28;
      }
      else if ( v1 > 0x4000 )
      {
        if ( v1 > 0x7000000000000000LL )
          goto LABEL_42;
        _BitScanReverse64((unsigned __int64 *)&v11, 2 * v1 - 1);
        if ( (unsigned __int64)(int)v11 < 7 )
          LOBYTE(v11) = 7;
        v12 = -(1LL << ((unsigned __int8)v11 - 3)) & (v1 + (1LL << ((unsigned __int8)v11 - 3)) - 1);
        if ( v1 > v12 || __CFADD__(unk_50607C0, v12) )
          goto LABEL_42;
LABEL_28:
        if ( v12 - 1 > 0x6FFFFFFFFFFFFFFFLL )
          goto LABEL_42;
LABEL_29:
        if ( *(char *)(v2 + 1) > 0 )
        {
          v13 = qword_50579C0[0];
          if ( qword_50579C0[0]
            || (v22 = unk_4F96994, v19 = sub_1300B80(v2, 0, (__int64)&off_49E8000), v10 = v22, (v13 = v19) != 0) )
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
        v21 = v10;
        v4 = (void *)sub_1318040(v2, v13, v12, 4096, v10, v14);
        if ( v4 )
        {
          LOBYTE(v23) = 1;
          v15 = v21;
          *(_QWORD *)&v24 = v2 + 824;
          *((_QWORD *)&v24 + 1) = v2 + 8;
          v25 = v2 + 16;
          v26 = v2 + 832;
          v16 = *(_QWORD *)(v2 + 824);
          *(_QWORD *)(v2 + 824) = v12 + v16;
          if ( v12 >= *(_QWORD *)(v2 + 16) - v16 )
          {
            sub_13133F0(v2, &v23);
            v15 = v21;
          }
          v17 = v4;
          if ( !v15 && unk_4F969A2 )
          {
            off_4C6F0B8(v4, v12);
            v17 = v4;
          }
          goto LABEL_37;
        }
LABEL_42:
        v17 = 0;
        v4 = 0;
LABEL_37:
        v23 = a1;
        v24 = 0;
        sub_1346E80(6, v4, v17, &v23);
        return v4;
      }
      v12 = 0x4000;
      if ( (unsigned __int64)(unk_50607C0 + 0x4000LL) <= 0x3FFF )
        goto LABEL_42;
      goto LABEL_29;
    }
  }
  if ( v1 > 0x3800 )
  {
    if ( v1 <= 0x4000 )
      goto LABEL_6;
    if ( v1 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v7, 2 * v1 - 1);
    if ( (unsigned __int64)(int)v7 < 7 )
      LOBYTE(v7) = 7;
    v3 = -(1LL << ((unsigned __int8)v7 - 3)) & (v1 + (1LL << ((unsigned __int8)v7 - 3)) - 1);
    if ( v1 > v3 || __CFADD__(unk_50607C0, v3) )
      return 0;
LABEL_17:
    if ( v3 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    goto LABEL_7;
  }
  if ( v1 > 0x1000 )
  {
    _BitScanReverse64(&v18, 2 * v1 - 1);
    v3 = -(1LL << ((unsigned __int8)v18 - 3)) & (v1 + (1LL << ((unsigned __int8)v18 - 3)) - 1);
  }
  else
  {
    v3 = qword_505FA40[byte_5060800[v1 >> 3]];
  }
  if ( v3 <= 0x3FFF )
    goto LABEL_17;
LABEL_6:
  v3 = 0x4000;
  if ( (unsigned __int64)(unk_50607C0 + 0x4000LL) <= 0x3FFF )
    return 0;
LABEL_7:
  v4 = (void *)sub_1318040(v2, 0, v3, 4096, 0, v2 + 856);
  if ( !v4 )
    return 0;
  LOBYTE(v23) = 1;
  *(_QWORD *)&v24 = v2 + 824;
  *((_QWORD *)&v24 + 1) = v2 + 8;
  v25 = v2 + 16;
  v26 = v2 + 832;
  v5 = *(_QWORD *)(v2 + 824);
  *(_QWORD *)(v2 + 824) = v3 + v5;
  if ( v3 >= *(_QWORD *)(v2 + 16) - v5 )
    sub_13133F0(v2, &v23);
  return v4;
}
