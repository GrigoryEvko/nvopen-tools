// Function: sub_10FDD10
// Address: 0x10fdd10
//
__int64 __fastcall sub_10FDD10(__int64 a1, char a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax
  int v4; // r13d
  __int64 *v5; // r14
  unsigned int i; // r15d
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 *v9; // [rsp+0h] [rbp-40h]
  int v10; // [rsp+Ch] [rbp-34h]

  v2 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    if ( v2 == 75 )
      return *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
    return *(_QWORD *)(a1 + 8);
  }
  if ( v2 == 18 )
  {
    result = sub_10FDBE0(a1, a2);
    if ( result )
      return result;
    v2 = *(_BYTE *)a1;
  }
  if ( v2 == 5 )
  {
    if ( *(_WORD *)(a1 + 2) == 46 )
      return *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
  }
  else if ( v2 > 0x15u )
  {
    return *(_QWORD *)(a1 + 8);
  }
  result = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(result + 8) == 17 )
  {
    v4 = *(_DWORD *)(result + 32);
    if ( v4 )
    {
      v5 = 0;
      for ( i = 0; i != v4; ++i )
      {
        if ( (unsigned int)*(unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a1, i) - 12 > 1 )
        {
          v7 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, i);
          if ( !v7 )
            return *(_QWORD *)(a1 + 8);
          if ( *v7 != 18 )
            return *(_QWORD *)(a1 + 8);
          v8 = sub_10FDBE0((__int64)v7, a2);
          if ( !v8 )
            return *(_QWORD *)(a1 + 8);
          if ( v5 )
          {
            v9 = (__int64 *)v8;
            v10 = sub_BCB090(v8);
            if ( v10 > (int)sub_BCB090((__int64)v5) )
              v5 = v9;
          }
          else
          {
            v5 = (__int64 *)v8;
          }
        }
      }
      if ( !v5 )
        return *(_QWORD *)(a1 + 8);
      result = sub_BCDA70(v5, v4);
      if ( !result )
        return *(_QWORD *)(a1 + 8);
    }
  }
  return result;
}
