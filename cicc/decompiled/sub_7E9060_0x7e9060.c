// Function: sub_7E9060
// Address: 0x7e9060
//
__int64 __fastcall sub_7E9060(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _QWORD *v7; // r13

  result = sub_731770(*a1, 0, a3, a4, a5, a6);
  if ( (_DWORD)result )
  {
    *(_QWORD *)(*a1 + 16) = 0;
    v7 = sub_7E88C0((const __m128i *)*a1);
    if ( *a2 )
    {
      result = (__int64)sub_73DF90(*a2, (__int64 *)*a1);
      *a2 = result;
    }
    else
    {
      result = *a1;
      *a2 = *a1;
    }
    *a1 = (__int64)v7;
  }
  return result;
}
