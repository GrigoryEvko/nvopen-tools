// Function: sub_12F9F70
// Address: 0x12f9f70
//
__int64 __fastcall sub_12F9F70(
        unsigned __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned int a5)
{
  __int128 v5; // rax
  unsigned __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // r15
  unsigned __int64 v9; // r14
  __int64 v10; // r10
  unsigned __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r8
  __int128 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r15
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  _BOOL8 v20; // r8
  char v22; // [rsp+Ch] [rbp-64h]
  char v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  *((_QWORD *)&v5 + 1) = a3;
  v6 = *((_QWORD *)&v5 + 1);
  v7 = a5;
  v8 = a1 & 0xFFFFFFFFFFFFLL;
  *((_QWORD *)&v5 + 1) &= 0xFFFFFFFFFFFFuLL;
  v9 = a2;
  v10 = *((_QWORD *)&v5 + 1);
  v11 = a4;
  v12 = HIWORD(a1) & 0x7FFF;
  v13 = HIWORD(v6) & 0x7FFF;
  v14 = v12 - v13;
  if ( v12 == v13 )
  {
    if ( v12 != 0x7FFF )
    {
      *(_QWORD *)&v15 = a2;
      *((_QWORD *)&v15 + 1) = a1 & 0xFFFFFFFFFFFFLL;
      v5 = __PAIR128__(v10, a4) + v15;
      if ( !v12 )
        return v5;
      v20 = 0;
      v19 = *((_QWORD *)&v5 + 1) | 0x2000000000000LL;
      goto LABEL_12;
    }
    *(_QWORD *)&v5 = v8 | a4 | a2 | *((_QWORD *)&v5 + 1);
    if ( !(_QWORD)v5 )
      return v5;
LABEL_19:
    *(_QWORD *)&v5 = sub_12FBB40(a1, a2, v6, a4, v14, v7);
    return v5;
  }
  if ( v14 < 0 )
  {
    if ( v13 == 0x7FFF )
    {
      *(_QWORD *)&v5 = a4 | *((_QWORD *)&v5 + 1);
      if ( __PAIR128__(a4, *((unsigned __int64 *)&v5 + 1)) == 0 )
        return v5;
      goto LABEL_19;
    }
    if ( v12 )
    {
      v8 |= 0x1000000000000uLL;
    }
    else if ( !++v14 )
    {
      v12 = v13 - 1;
      v16 = 0;
      goto LABEL_10;
    }
    v22 = v7;
    v24 = *((_QWORD *)&v5 + 1);
    sub_12FB780(&v25, v8, a2, 0, -v14);
    v16 = v25;
    v9 = v26;
    v8 = v27;
    v10 = v24;
    LOBYTE(v7) = v22;
    v12 = v13 - 1;
    goto LABEL_10;
  }
  if ( v12 == 0x7FFF )
  {
    *(_QWORD *)&v5 = a2 | v8;
    if ( !(a2 | v8) )
      return v5;
    goto LABEL_19;
  }
  if ( v13 )
  {
    v10 = *((_QWORD *)&v5 + 1) | 0x1000000000000LL;
LABEL_9:
    v23 = v7;
    v13 = HIWORD(a1) & 0x7FFF;
    --v12;
    sub_12FB780(&v25, v10, a4, 0, v14);
    v16 = v25;
    v11 = v26;
    v10 = v27;
    LOBYTE(v7) = v23;
    goto LABEL_10;
  }
  if ( --v14 )
    goto LABEL_9;
  v12 = 0;
  v16 = 0;
  v13 = 1;
LABEL_10:
  *(_QWORD *)&v5 = v11 + v9;
  v17 = v8 | 0x1000000000000LL;
  v18 = v11 + v9;
  *((_QWORD *)&v5 + 1) = __CFADD__(v11, v9) + v17 + v10;
  if ( *((_QWORD *)&v5 + 1) > 0x1FFFFFFFFFFFFuLL )
  {
    v19 = __CFADD__(v11, v9) + v17 + v10;
    v12 = v13;
    v20 = v16 != 0;
LABEL_12:
    v16 = ((_QWORD)v5 << 63) | v20;
    *((_QWORD *)&v5 + 1) = v19 >> 1;
    v18 = ((unsigned __int64)v5 >> 1) | (v19 << 63);
  }
  *(_QWORD *)&v5 = sub_12FC4A0((unsigned __int8)v7, v12, *((_QWORD *)&v5 + 1), v18, v16);
  return v5;
}
