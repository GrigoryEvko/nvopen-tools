// Function: sub_AD7A80
// Address: 0xad7a80
//
bool __fastcall sub_AD7A80(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r12
  unsigned int v6; // ebx
  bool result; // al
  __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // r13
  unsigned int v11; // ebx
  bool v12; // [rsp+Fh] [rbp-31h]
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  v5 = a1;
  while ( 1 )
  {
    if ( *v5 == 17 )
    {
      v6 = *((_DWORD *)v5 + 8);
      if ( v6 <= 0x40 )
        return *((_QWORD *)v5 + 3) == 1;
      else
        return v6 - 1 == (unsigned int)sub_C444A0(v5 + 24);
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
  v10 = v13;
  v11 = v14;
  result = v13 == 1;
  if ( v14 > 0x40 )
  {
    result = v11 - 1 == (unsigned int)sub_C444A0(&v13);
    if ( v10 )
    {
      v12 = result;
      j_j___libc_free_0_0(v10);
      return v12;
    }
  }
  return result;
}
