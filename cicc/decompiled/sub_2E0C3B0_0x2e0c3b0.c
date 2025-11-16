// Function: sub_2E0C3B0
// Address: 0x2e0c3b0
//
_QWORD *__fastcall sub_2E0C3B0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *result; // rax
  unsigned int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rsi
  __m128i v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+10h] [rbp-40h]

  result = (_QWORD *)sub_2E09D00((__int64 *)a1, a2);
  v8 = *(_DWORD *)(a1 + 8);
  v9 = *(_QWORD *)a1 + 24LL * v8;
  if ( result != (_QWORD *)v9 )
  {
    v10 = result[2];
    v11 = result[1];
    if ( *result == a2 )
    {
      if ( a3 == v11 )
      {
        if ( (_QWORD *)v9 != result + 3 )
        {
          result = memmove(result, result + 3, v9 - (_QWORD)(result + 3));
          v8 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v8 - 1;
        if ( a4 )
          return sub_2E0A490(a1, v10);
      }
      else
      {
        *result = a3;
      }
    }
    else
    {
      result[1] = a2;
      if ( a3 != v11 )
      {
        v12.m128i_i64[1] = v11;
        v12.m128i_i64[0] = a3;
        v13 = v10;
        return (_QWORD *)sub_2E0C1A0(a1, (__m128i *)(result + 3), &v12);
      }
    }
  }
  return result;
}
