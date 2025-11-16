// Function: sub_12FBEC0
// Address: 0x12fbec0
//
unsigned __int64 __fastcall sub_12FBEC0(char a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, char a5)
{
  unsigned __int64 result; // rax
  int v6; // r13d
  char v7; // r15
  __int64 v8; // r12
  __int64 v9; // r14
  unsigned __int64 v10; // rdx
  bool v11; // r14
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  bool v16; // cf
  unsigned __int64 v17; // rax
  int v18; // r8d
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // r12
  int v22; // edx
  unsigned __int64 v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-38h]
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]

  result = a3;
  v6 = unk_4F968EB;
  v7 = unk_4F968EB & 0xFB;
  if ( a5 == 80 )
    goto LABEL_4;
  if ( a5 == 64 )
  {
    v8 = 2047;
    v9 = 1024;
    goto LABEL_17;
  }
  v8 = 0xFFFFFFFFFFLL;
  v9 = 0x8000000000LL;
  if ( a5 == 32 )
  {
LABEL_17:
    v13 = (a4 != 0) | a3;
    if ( v7 )
    {
      v9 = 0;
      if ( unk_4F968EB == (a1 == 0) + 2 )
        v9 = v8;
    }
    if ( (unsigned int)(a2 - 1) <= 0x7FFC )
    {
LABEL_21:
      v14 = v8 + 1;
      v15 = v13 & v8;
      if ( (v13 & v8) != 0 )
      {
        unk_4F968EA |= 1u;
        if ( unk_4F968EB == 6 )
          return v14 | ~v8 & v13;
      }
      v16 = __CFADD__(v9, v13);
      v17 = v9 + v13;
      if ( v16 )
        v17 = 0x8000000000000000LL;
      if ( !unk_4F968EB && 2 * v15 == v14 )
        v8 |= v14;
      return ~v8 & v17;
    }
    if ( a2 > 0 )
    {
      if ( a2 > 32766 || a2 == 32766 && __CFADD__(v13, v9) )
        goto LABEL_68;
      goto LABEL_21;
    }
    if ( unk_4C6F00D && !a2 )
    {
      v23 = (v13 >> (1 - (unsigned __int8)a2)) | (v13 << ((unsigned __int8)a2 - 1) != 0);
      v24 = v23 & v8;
      if ( (v23 & v8) != 0 )
      {
        if ( v9 + v13 < v13 )
          goto LABEL_58;
        goto LABEL_66;
      }
LABEL_75:
      v17 = v9 + v23;
      return ~v8 & v17;
    }
    if ( 1 - a2 <= 62 )
    {
      v23 = (v13 >> (1 - (unsigned __int8)a2)) | (v13 << ((unsigned __int8)a2 - 1) != 0);
      v24 = v23 & v8;
      if ( (v23 & v8) == 0 )
        goto LABEL_75;
    }
    else
    {
      v23 = v13 != 0;
      v24 = v23 & v8;
      if ( (v23 & v8) == 0 )
        goto LABEL_75;
      v23 &= v8;
    }
LABEL_66:
    v27 = v24;
    sub_12F9B70(2);
    v24 = v27;
LABEL_58:
    v25 = v8 + 1;
    unk_4F968EA |= 1u;
    if ( (_BYTE)v6 == 6 )
    {
      v17 = v9 + (v23 | v25);
    }
    else
    {
      v17 = v9 + v23;
      if ( !(_BYTE)v6 )
      {
        v26 = 2 * v24;
        if ( v26 == v25 )
          v8 |= v26;
      }
    }
    return ~v8 & v17;
  }
LABEL_4:
  if ( v7 )
    LOBYTE(v10) = a4 != 0 && unk_4F968EB == (a1 == 0) + 2;
  else
    v10 = a4 >> 63;
  v11 = unk_4F968EB == 0;
  if ( (unsigned int)(a2 - 1) <= 0x7FFC )
    goto LABEL_11;
  if ( a2 <= 0 )
  {
    v18 = 1;
    if ( unk_4C6F00D )
      v18 = (result != -1) | (a2 < 0) | (unsigned __int8)v10 ^ 1;
    v19 = 1 - a2;
    if ( 1 - a2 <= 63 )
    {
      v20 = result << ((unsigned __int8)a2 - 1);
      result >>= v19;
    }
    else
    {
      if ( v19 == 64 )
        v20 = result;
      else
        v20 = result != 0;
      result = 0;
    }
    v21 = v20 | (a4 != 0);
    if ( v21 )
    {
      if ( v18 )
      {
        v28 = result;
        sub_12F9B70(2);
        result = v28;
      }
      unk_4F968EA |= 1u;
      if ( (_BYTE)v6 == 6 )
      {
        result |= 1u;
        return result;
      }
      if ( !v7 )
      {
        if ( v21 >= 0 )
          return result;
        return (result + 1) & ~(unsigned __int64)(v11 & (unsigned __int8)((v21 & 0x7FFFFFFFFFFFFFFFLL) == 0));
      }
      v22 = 2;
      if ( a1 )
      {
LABEL_50:
        if ( v22 != v6 || !v21 )
          return result;
        return (result + 1) & ~(unsigned __int64)(v11 & (unsigned __int8)((v21 & 0x7FFFFFFFFFFFFFFFLL) == 0));
      }
    }
    else if ( !v7 || a1 )
    {
      return result;
    }
    v22 = 3;
    goto LABEL_50;
  }
  if ( a2 <= 32766 && (result != -1 || a2 != 32766 || !(_BYTE)v10) )
  {
LABEL_11:
    if ( a4 && (unk_4F968EA |= 1u, unk_4F968EB == 6) )
    {
      result |= 1u;
    }
    else if ( (_BYTE)v10 )
    {
      v12 = result + 1;
      if ( v12 )
        return ~(unsigned __int64)(((a4 & 0x7FFFFFFFFFFFFFFFLL) == 0) & (unsigned __int8)v11) & v12;
      else
        return 0x8000000000000000LL;
    }
    return result;
  }
  v8 = 0;
LABEL_68:
  sub_12F9B70(5);
  result = 0x8000000000000000LL;
  if ( v7 && v6 != (a1 == 0) + 2 )
    return ~v8;
  return result;
}
