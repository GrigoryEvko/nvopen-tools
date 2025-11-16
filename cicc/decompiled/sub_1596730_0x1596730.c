// Function: sub_1596730
// Address: 0x1596730
//
__int64 __fastcall sub_1596730(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 i; // r12
  char v6; // al
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned int v9; // ebx
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
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned int v24; // r8d
  __int64 v25; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v28[7]; // [rsp+18h] [rbp-38h] BYREF

  for ( i = a1; ; i = v15 )
  {
    v6 = *(_BYTE *)(i + 16);
    if ( v6 == 13 )
      break;
    if ( v6 == 14 )
    {
      v11 = sub_16982C0(a1, a2, a3, a4);
      v12 = i + 32;
      if ( *(_QWORD *)(i + 32) == v11 )
        sub_169D930(&v27, v12);
      else
        sub_169D7E0(&v27, v12);
      v13 = v27;
      v14 = LODWORD(v28[0]) - 1;
      if ( LODWORD(v28[0]) <= 0x40 )
      {
        LOBYTE(v4) = v27 != 1LL << v14;
      }
      else
      {
        v4 = 1;
        if ( (*(_QWORD *)(v27 + 8LL * (v14 >> 6)) & (1LL << v14)) != 0 )
          LOBYTE(v4) = (unsigned int)sub_16A58A0(&v27) != v14;
        if ( v13 )
          j_j___libc_free_0_0(v13);
      }
      return v4;
    }
    if ( v6 != 8 )
      goto LABEL_20;
    a1 = i;
    v15 = sub_1594B20(i);
    if ( !v15 )
    {
      v6 = *(_BYTE *)(i + 16);
LABEL_20:
      if ( v6 == 12 && (unsigned __int8)sub_1595CF0(i) )
      {
        if ( (unsigned __int8)(*(_BYTE *)(sub_1595890(i) + 8) - 1) > 5u )
        {
          sub_1595AB0((__int64)&v27, i, 0);
          v4 = sub_13CFF40(&v27, i, v22, v23, v24) ^ 1;
          if ( LODWORD(v28[0]) > 0x40 && v27 )
            j_j___libc_free_0_0(v27);
        }
        else
        {
          sub_1595B70((__int64)&v27, i, 0);
          v18 = sub_16982C0(&v27, i, v16, v17);
          if ( v28[0] == v18 )
            sub_169D930(&v25, v28);
          else
            sub_169D7E0(&v25, v28);
          v4 = sub_13CFF40(&v25, (__int64)v28, v19, v20, v21) ^ 1;
          if ( v26 > 0x40 && v25 )
            j_j___libc_free_0_0(v25);
          sub_127D120(v28);
        }
      }
      else
      {
        return 0;
      }
      return v4;
    }
  }
  v7 = *(_DWORD *)(i + 32);
  v8 = *(_QWORD *)(i + 24);
  v9 = v7 - 1;
  if ( v7 > 0x40 )
  {
    v4 = 1;
    if ( (*(_QWORD *)(v8 + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
      LOBYTE(v4) = (unsigned int)sub_16A58A0(i + 24) != v9;
    return v4;
  }
  LOBYTE(v4) = v8 != 1LL << v9;
  return v4;
}
