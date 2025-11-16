// Function: sub_73E000
// Address: 0x73e000
//
_BYTE *__fastcall sub_73E000(__int64 *a1, __m128i *a2, const __m128i **a3, __int64 a4, _DWORD *a5)
{
  __int64 v7; // r12
  _BYTE *result; // rax
  _DWORD v10[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( *a1 )
  {
    v7 = *(_QWORD *)*a1;
  }
  else
  {
    if ( *a3 )
      v7 = (*a3)[8].m128i_i64[0];
    else
      v7 = a2[8].m128i_i64[0];
    if ( (unsigned int)sub_8D32E0(v7) )
      v7 = sub_8D46C0(v7);
  }
  while ( *(_BYTE *)(v7 + 140) == 12 )
    v7 = *(_QWORD *)(v7 + 160);
  result = (_BYTE *)sub_8DAAE0(v7, a4);
  if ( !(_DWORD)result )
  {
    if ( *a1 )
    {
      result = sub_73DBF0(5u, a4, *a1);
      *a1 = (__int64)result;
      result[27] |= 2u;
    }
    else
    {
      if ( *a3 )
      {
        sub_72A510(*a3, a2);
        *a3 = 0;
      }
      return (_BYTE *)sub_712540(a2, a4, 1, 0, v10, a5);
    }
  }
  return result;
}
