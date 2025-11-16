// Function: sub_15964D0
// Address: 0x15964d0
//
bool __fastcall sub_15964D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r12
  char v5; // al
  unsigned int v6; // eax
  __int64 v7; // rsi
  unsigned int v8; // ebx
  int v9; // r8d
  bool result; // al
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // r8d
  char v22; // al
  bool v23; // [rsp+Fh] [rbp-51h]
  char v24; // [rsp+Fh] [rbp-51h]
  char v25; // [rsp+Fh] [rbp-51h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v29[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = a1; ; i = v15 )
  {
    v5 = *(_BYTE *)(i + 16);
    if ( v5 == 13 )
      break;
    if ( v5 == 14 )
    {
      v11 = sub_16982C0(a1, a2, a3, a4);
      v12 = i + 32;
      if ( *(_QWORD *)(i + 32) == v11 )
        sub_169D930(&v28, v12);
      else
        sub_169D7E0(&v28, v12);
      v13 = v28;
      v14 = LODWORD(v29[0]) - 1;
      if ( LODWORD(v29[0]) <= 0x40 )
        return v28 == 1LL << v14;
      result = 0;
      if ( (*(_QWORD *)(v28 + 8LL * (v14 >> 6)) & (1LL << v14)) != 0 )
        goto LABEL_14;
      goto LABEL_15;
    }
    if ( v5 != 8 )
      goto LABEL_21;
    a1 = i;
    v15 = sub_1594B20(i);
    if ( !v15 )
    {
      v5 = *(_BYTE *)(i + 16);
LABEL_21:
      if ( v5 == 12 && (unsigned __int8)sub_1595CF0(i) )
      {
        if ( (unsigned __int8)(*(_BYTE *)(sub_1595890(i) + 8) - 1) <= 5u )
        {
          sub_1595B70((__int64)&v28, i, 0);
          v18 = sub_16982C0(&v28, i, v16, v17);
          if ( v29[0] == v18 )
            sub_169D930(&v26, v29);
          else
            sub_169D7E0(&v26, v29);
          v22 = sub_13CFF40(&v26, (__int64)v29, v19, v20, v21);
          if ( v27 > 0x40 && v26 )
          {
            v24 = v22;
            j_j___libc_free_0_0(v26);
            v22 = v24;
          }
          v25 = v22;
          sub_127D120(v29);
          return v25;
        }
        sub_1595AB0((__int64)&v28, i, 0);
        if ( LODWORD(v29[0]) <= 0x40 )
          return 1LL << (LOBYTE(v29[0]) - 1) == v28;
        v14 = LODWORD(v29[0]) - 1;
        v13 = v28;
        result = 0;
        if ( (*(_QWORD *)(v28 + 8LL * ((unsigned int)(LODWORD(v29[0]) - 1) >> 6)) & (1LL << (LOBYTE(v29[0]) - 1))) != 0 )
LABEL_14:
          result = (unsigned int)sub_16A58A0(&v28) == v14;
LABEL_15:
        if ( v13 )
        {
          v23 = result;
          j_j___libc_free_0_0(v13);
          return v23;
        }
        return result;
      }
      return 0;
    }
  }
  v6 = *(_DWORD *)(i + 32);
  v7 = *(_QWORD *)(i + 24);
  v8 = v6 - 1;
  if ( v6 <= 0x40 )
    return v7 == 1LL << v8;
  if ( (*(_QWORD *)(v7 + 8LL * (v8 >> 6)) & (1LL << v8)) == 0 )
    return 0;
  v9 = sub_16A58A0(i + 24);
  result = 1;
  if ( v9 != v8 )
    return 0;
  return result;
}
