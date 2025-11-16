// Function: sub_AD7930
// Address: 0xad7930
//
bool __fastcall sub_AD7930(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r12
  unsigned int v6; // ebx
  bool result; // al
  __int64 v8; // rax
  _BYTE *v9; // rsi
  unsigned int v10; // ebx
  __int64 v11; // r13
  bool v12; // [rsp+Fh] [rbp-31h]
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  v5 = a1;
  while ( 1 )
  {
    if ( *v5 == 17 )
    {
      v6 = *((_DWORD *)v5 + 8);
      result = 1;
      if ( v6 )
      {
        if ( v6 <= 0x40 )
          return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6) == *((_QWORD *)v5 + 3);
        else
          return v6 == (unsigned int)sub_C445E0(v5 + 24);
      }
      return result;
    }
    if ( *v5 == 18 )
      break;
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v5 + 1) + 8LL) - 17 <= 1 )
    {
      a1 = v5;
      a2 = 0;
      v5 = sub_AD7630((__int64)v5, 0, a3);
      if ( v5 )
        continue;
    }
    return 0;
  }
  v8 = sub_C33340(a1, a2, a3, a4, a5);
  v9 = v5 + 24;
  if ( *((_QWORD *)v5 + 3) == v8 )
    sub_C3E660(&v13, v9);
  else
    sub_C3A850(&v13, v9);
  v10 = v14;
  result = 1;
  if ( v14 )
  {
    v11 = v13;
    if ( v14 <= 0x40 )
    {
      return v13 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14);
    }
    else
    {
      result = v10 == (unsigned int)sub_C445E0(&v13);
      if ( v11 )
      {
        v12 = result;
        j_j___libc_free_0_0(v11);
        return v12;
      }
    }
  }
  return result;
}
