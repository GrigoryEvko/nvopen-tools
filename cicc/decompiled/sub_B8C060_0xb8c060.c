// Function: sub_B8C060
// Address: 0xb8c060
//
_DWORD *__fastcall sub_B8C060(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  _DWORD *result; // rax
  _DWORD *v4; // r13
  _DWORD *v5; // rbx
  _BYTE *v6; // rsi

  v1 = a1[24];
  if ( v1 != a1[25] )
    a1[25] = v1;
  v2 = a1[17];
  if ( v2 != a1[18] )
    a1[18] = v2;
  result = (_DWORD *)a1[20];
  v4 = (_DWORD *)a1[21];
  v5 = result + 2;
  if ( result != v4 )
  {
    while ( 1 )
    {
      v6 = (_BYTE *)a1[18];
      if ( v6 == (_BYTE *)a1[19] )
      {
        sub_B8BD80((__int64)(a1 + 17), v6, v5);
        result = v5 + 4;
        if ( v4 == v5 + 2 )
          return result;
      }
      else
      {
        if ( v6 )
        {
          *(_DWORD *)v6 = *v5;
          v6 = (_BYTE *)a1[18];
        }
        result = v5 + 4;
        a1[18] = v6 + 4;
        if ( v4 == v5 + 2 )
          return result;
      }
      v5 = result;
    }
  }
  return result;
}
