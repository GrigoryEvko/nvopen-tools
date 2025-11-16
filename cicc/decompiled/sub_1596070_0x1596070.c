// Function: sub_1596070
// Address: 0x1596070
//
bool __fastcall sub_1596070(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r12
  char v5; // al
  unsigned int v6; // ebx
  bool result; // al
  __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // ebx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // ebx
  __int64 v17; // r14
  bool v18; // al
  int v19; // ebx
  bool v20; // [rsp+Fh] [rbp-51h]
  bool v21; // [rsp+Fh] [rbp-51h]
  bool v22; // [rsp+Fh] [rbp-51h]
  bool v23; // [rsp+Fh] [rbp-51h]
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v27[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = a1; ; i = v12 )
  {
    v5 = *(_BYTE *)(i + 16);
    if ( v5 == 13 )
    {
      v6 = *(_DWORD *)(i + 32);
      if ( v6 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == *(_QWORD *)(i + 24);
      else
        return v6 == (unsigned int)sub_16A58F0(i + 24);
    }
    if ( v5 == 14 )
    {
      v8 = sub_16982C0(a1, a2, a3, a4);
      v9 = i + 32;
      if ( *(_QWORD *)(i + 32) == v8 )
        sub_169D930(&v26, v9);
      else
        sub_169D7E0(&v26, v9);
      v10 = v27[0];
      v11 = v26;
      if ( LODWORD(v27[0]) <= 0x40 )
        return v26 == 0xFFFFFFFFFFFFFFFFLL >> (64 - LOBYTE(v27[0]));
      result = v10 == (unsigned int)sub_16A58F0(&v26);
      if ( v11 )
      {
        v20 = result;
        j_j___libc_free_0_0(v11);
        return v20;
      }
      return result;
    }
    if ( v5 != 8 )
      break;
    a1 = i;
    v12 = sub_1594B20(i);
    if ( !v12 )
    {
      v5 = *(_BYTE *)(i + 16);
      break;
    }
  }
  if ( v5 != 12 || !(unsigned __int8)sub_1595CF0(i) )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)(sub_1595890(i) + 8) - 1) > 5u )
  {
    sub_1595AB0((__int64)&v26, i, 0);
    v19 = v27[0];
    if ( LODWORD(v27[0]) <= 0x40 )
    {
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - LOBYTE(v27[0])) == v26;
    }
    else
    {
      result = v19 == (unsigned int)sub_16A58F0(&v26);
      if ( v26 )
      {
        v23 = result;
        j_j___libc_free_0_0(v26);
        return v23;
      }
    }
  }
  else
  {
    sub_1595B70((__int64)&v26, i, 0);
    v15 = sub_16982C0(&v26, i, v13, v14);
    if ( v27[0] == v15 )
      sub_169D930(&v24, v27);
    else
      sub_169D7E0(&v24, v27);
    v16 = v25;
    v17 = v24;
    if ( v25 <= 0x40 )
    {
      v18 = v24 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25);
    }
    else
    {
      v18 = v16 == (unsigned int)sub_16A58F0(&v24);
      if ( v17 )
      {
        v21 = v18;
        j_j___libc_free_0_0(v17);
        v18 = v21;
      }
    }
    v22 = v18;
    sub_127D120(v27);
    return v22;
  }
  return result;
}
