// Function: sub_12FA1E0
// Address: 0x12fa1e0
//
__int64 __fastcall sub_12FA1E0(
        unsigned __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5)
{
  __int64 result; // rax
  unsigned __int8 v8; // cl
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rbx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r13
  __int64 v15; // rdx
  unsigned __int128 v16; // kr00_16
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int8 v21; // [rsp+7h] [rbp-39h]
  char v22; // [rsp+8h] [rbp-38h]

  result = a2;
  v8 = a5;
  v9 = 16 * a2;
  v10 = 16 * a4;
  v11 = HIWORD(a3) & 0x7FFF;
  v12 = HIWORD(a1) & 0x7FFF;
  v13 = (a2 >> 60) | (16 * a1) & 0xFFFFFFFFFFFF0LL;
  v14 = (a4 >> 60) | (16 * a3) & 0xFFFFFFFFFFFF0LL;
  v15 = v12 - v11;
  if ( v12 - v11 > 0 )
  {
    if ( v12 == 0x7FFF )
    {
      if ( !(v9 | v13) )
        return result;
      return sub_12FBB40(a1, a2, a3, a4, a5, v9);
    }
    if ( v11 )
    {
      v14 |= 0x10000000000000uLL;
    }
    else if ( v15 == 1 )
    {
LABEL_15:
      v13 |= 0x10000000000000uLL;
      goto LABEL_16;
    }
    v21 = a5;
    v17 = sub_12FB6F0(v14, 16 * a4);
    v8 = v21;
    v9 = 16 * a2;
    v10 = v17;
    v14 = v18;
    goto LABEL_15;
  }
  if ( (HIWORD(a1) & 0x7FFF) != v11 )
  {
    if ( v11 == 0x7FFF )
    {
      if ( !(v10 | v14) )
        return 0;
      return sub_12FBB40(a1, a2, a3, a4, a5, v9);
    }
    if ( v12 )
    {
      v13 |= 0x10000000000000uLL;
    }
    else if ( v15 == -1 )
    {
LABEL_29:
      v14 |= 0x10000000000000uLL;
      v12 = v11;
      goto LABEL_10;
    }
    v22 = a5;
    v19 = sub_12FB6F0(v13, v9);
    LOBYTE(a5) = v22;
    v9 = v19;
    v13 = v20;
    goto LABEL_29;
  }
  if ( v12 == 0x7FFF )
  {
    v9 |= v10;
    if ( !(v13 | v9 | v14) )
    {
      sub_12F9B70(16);
      return 0;
    }
    return sub_12FBB40(a1, a2, a3, a4, a5, v9);
  }
  if ( !v12 )
    v12 = 1;
  if ( v14 < v13 )
    goto LABEL_16;
  if ( v14 > v13 )
  {
LABEL_10:
    v8 = a5 ^ 1;
    v16 = __PAIR128__(v14 - v13, v10) - v9;
    return sub_12FC7A0(v8, v12 - 5, *((_QWORD *)&v16 + 1), v16);
  }
  if ( v10 < v9 )
  {
LABEL_16:
    v16 = __PAIR128__(v13 - v14, v9) - v10;
    return sub_12FC7A0(v8, v12 - 5, *((_QWORD *)&v16 + 1), v16);
  }
  if ( v10 > v9 )
    goto LABEL_10;
  return 0;
}
