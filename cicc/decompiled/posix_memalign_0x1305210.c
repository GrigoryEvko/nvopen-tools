// Function: posix_memalign
// Address: 0x1305210
//
__int64 __fastcall posix_memalign(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v6; // r13
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rdx
  unsigned int v12; // r13d
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // r15
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r10
  unsigned __int8 v20; // r11
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rax
  size_t v24; // r10
  unsigned __int8 v25; // r11
  void *v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rcx
  void *v32; // [rsp+0h] [rbp-80h]
  unsigned __int8 v33; // [rsp+10h] [rbp-70h]
  unsigned __int64 v34; // [rsp+10h] [rbp-70h]
  size_t v35; // [rsp+18h] [rbp-68h]
  unsigned __int8 v36; // [rsp+18h] [rbp-68h]
  void *v37; // [rsp+18h] [rbp-68h]
  __int64 *v38; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v39; // [rsp+28h] [rbp-58h]
  unsigned __int64 v40; // [rsp+30h] [rbp-50h]
  __int64 v41; // [rsp+38h] [rbp-48h]
  __int64 v42; // [rsp+40h] [rbp-40h]

  v6 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v15 = v6;
    v6 = sub_1313D30(v6, 0);
    if ( *(_BYTE *)(v6 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v15, 0) )
      {
        *__errno_location() = 12;
        *a1 = 0;
        return 12;
      }
      if ( a2 <= 7 || (v16 = (a2 - 1) & a2) != 0 )
      {
        v26 = (void *)*a1;
        v16 = 22;
        v12 = 22;
        goto LABEL_49;
      }
      v17 = 1;
      if ( a3 )
        v17 = a3;
      if ( a3 > 0x3800 || a2 > 0x1000 )
      {
        if ( a2 > 0x7000000000000000LL )
          goto LABEL_58;
        if ( a3 <= 0x4000 )
        {
LABEL_38:
          v19 = 0x4000;
          if ( ((a2 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
            goto LABEL_58;
          goto LABEL_39;
        }
        if ( a3 > 0x7000000000000000LL )
          goto LABEL_58;
        _BitScanReverse64((unsigned __int64 *)&v28, 2 * v17 - 1);
        if ( (unsigned __int64)(int)v28 < 7 )
          LOBYTE(v28) = 7;
        v19 = -(1LL << ((unsigned __int8)v28 - 3)) & ((1LL << ((unsigned __int8)v28 - 3)) + v17 - 1);
        if ( v19 < v17 || __CFADD__(v19, unk_50607C0 + ((a2 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
          goto LABEL_58;
      }
      else
      {
        v18 = -(__int64)a2 & (v17 + a2 - 1);
        if ( v18 > 0x1000 )
        {
          _BitScanReverse64(&v31, 2 * v18 - 1);
          v19 = -(1LL << ((unsigned __int8)v31 - 3)) & (v18 + (1LL << ((unsigned __int8)v31 - 3)) - 1);
        }
        else
        {
          v19 = qword_505FA40[byte_5060800[(v18 + 7) >> 3]];
        }
        if ( v19 > 0x3FFF )
          goto LABEL_38;
      }
      if ( v19 - 1 > 0x6FFFFFFFFFFFFFFFLL )
        goto LABEL_58;
LABEL_39:
      v20 = unk_4F96994;
      if ( *(char *)(v6 + 1) <= 0 )
      {
        if ( *(_BYTE *)v6 )
        {
          v21 = v6 + 856;
          v22 = 0;
          goto LABEL_42;
        }
        goto LABEL_62;
      }
      v21 = 0;
      v22 = qword_50579C0[0];
      if ( qword_50579C0[0]
        || (v34 = v19,
            v36 = unk_4F96994,
            v29 = sub_1300B80(v6, 0, (__int64)&off_49E8000),
            v20 = v36,
            v19 = v34,
            v21 = 0,
            (v22 = v29) != 0) )
      {
LABEL_42:
        v33 = v20;
        v35 = v19;
        v23 = sub_1318040(v6, v22, v19, a2, v20, v21);
        v24 = v35;
        v25 = v33;
        v26 = (void *)v23;
        if ( v23 )
        {
          LOBYTE(v38) = 1;
          v39 = v6 + 824;
          v40 = v6 + 8;
          v41 = v6 + 16;
          v42 = v6 + 832;
          v27 = *(_QWORD *)(v6 + 824);
          *(_QWORD *)(v6 + 824) = v27 + v35;
          if ( *(_QWORD *)(v6 + 16) - v27 <= v35 )
          {
            v32 = (void *)v23;
            sub_13133F0(v6, &v38);
            v26 = v32;
            v24 = v35;
            v25 = v33;
          }
          if ( !v25 && unk_4F969A2 )
          {
            v37 = v26;
            off_4C6F0B8(v26, v24);
            v26 = v37;
          }
          *a1 = (__int64)v26;
          v12 = 0;
          goto LABEL_49;
        }
        goto LABEL_58;
      }
      if ( unk_505F9B8 )
      {
LABEL_62:
        v21 = 0;
        v22 = 0;
        goto LABEL_42;
      }
LABEL_58:
      v26 = (void *)*a1;
      v16 = 12;
      v12 = 12;
LABEL_49:
      v38 = a1;
      v39 = a2;
      v40 = a3;
      sub_1346E80(1, v26, v16, &v38);
      return v12;
    }
  }
  if ( a2 <= 7 || ((a2 - 1) & a2) != 0 )
    return 22;
  v7 = 1;
  if ( a3 )
    v7 = a3;
  if ( a2 <= 0x1000 && a3 <= 0x3800 )
  {
    v8 = -(__int64)a2 & (v7 + a2 - 1);
    if ( v8 > 0x1000 )
    {
      _BitScanReverse64(&v30, 2 * v8 - 1);
      v9 = (v8 + (1LL << ((unsigned __int8)v30 - 3)) - 1) & -(1LL << ((unsigned __int8)v30 - 3));
    }
    else
    {
      v9 = qword_505FA40[byte_5060800[(v8 + 7) >> 3]];
    }
    if ( v9 > 0x3FFF )
      goto LABEL_11;
LABEL_24:
    if ( v9 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 12;
    goto LABEL_12;
  }
  if ( a2 > 0x7000000000000000LL )
    return 12;
  if ( a3 > 0x4000 )
  {
    if ( a3 > 0x7000000000000000LL )
      return 12;
    _BitScanReverse64((unsigned __int64 *)&v14, 2 * v7 - 1);
    if ( (unsigned __int64)(int)v14 < 7 )
      LOBYTE(v14) = 7;
    v9 = -(1LL << ((unsigned __int8)v14 - 3)) & ((1LL << ((unsigned __int8)v14 - 3)) + v7 - 1);
    if ( v9 < v7 || __CFADD__(v9, unk_50607C0 + ((a2 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
      return 12;
    goto LABEL_24;
  }
LABEL_11:
  v9 = 0x4000;
  if ( unk_50607C0 + ((a2 + 4095) & 0xFFFFFFFFFFFFF000LL) + 12288 <= 0x3FFF )
    return 12;
LABEL_12:
  v10 = sub_1318040(v6, 0, v9, a2, 0, v6 + 856);
  if ( !v10 )
    return 12;
  LOBYTE(v38) = 1;
  v39 = v6 + 824;
  v40 = v6 + 8;
  v41 = v6 + 16;
  v42 = v6 + 832;
  v11 = *(_QWORD *)(v6 + 824);
  *(_QWORD *)(v6 + 824) = v9 + v11;
  if ( v9 >= *(_QWORD *)(v6 + 16) - v11 )
    sub_13133F0(v6, &v38);
  *a1 = v10;
  return 0;
}
