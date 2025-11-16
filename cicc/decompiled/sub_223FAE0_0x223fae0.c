// Function: sub_223FAE0
// Address: 0x223fae0
//
__int64 __fastcall sub_223FAE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rdx

  result = 0x80000000LL;
  a1[4] = a2;
  a1[6] = a3;
  if ( a4 >= 0x80000000LL )
  {
    v5 = a2 + a4 + 0x7FFFFFFF;
    do
    {
      a2 = v5 - a4;
      a4 -= 0x7FFFFFFF;
    }
    while ( a4 >= 0x80000000LL );
  }
  a1[5] = a2 + (int)a4;
  return result;
}
