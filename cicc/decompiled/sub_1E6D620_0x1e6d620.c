// Function: sub_1E6D620
// Address: 0x1e6d620
//
__int64 *__fastcall sub_1E6D620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 i; // rbx
  __int64 *result; // rax

  v6 = a2 - a1;
  if ( v6 > 8 )
  {
    for ( i = ((v6 >> 3) - 2) / 2; ; --i )
    {
      result = sub_1E6D510(a1, i, v6 >> 3, *(_QWORD *)(a1 + 8 * i), a5, a6, *(_OWORD *)a3, *(_QWORD *)(a3 + 16));
      if ( !i )
        break;
    }
  }
  return result;
}
