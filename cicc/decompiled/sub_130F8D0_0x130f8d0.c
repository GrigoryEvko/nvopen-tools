// Function: sub_130F8D0
// Address: 0x130f8d0
//
__int64 __fastcall sub_130F8D0(__int64 a1, __int64 a2)
{
  signed __int64 v2; // rcx
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rdx
  bool v6; // zf
  __int64 result; // rax

  v2 = qword_4F96A00;
  do
  {
    v4 = a2 + v2;
    v5 = a2 + v2;
    if ( qword_4F96A08 <= (unsigned __int64)(a2 + v2) )
      v5 = v4 % qword_4F96A08;
    result = _InterlockedCompareExchange64(&qword_4F96A00, v5, v2);
    v6 = v2 == result;
    v2 = result;
  }
  while ( !v6 );
  if ( qword_4F96A08 <= v4 )
    return sub_1308810(0, 0, (__int64)byte_4F96A10);
  return result;
}
