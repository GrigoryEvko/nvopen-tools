// Function: sub_1623BA0
// Address: 0x1623ba0
//
unsigned __int64 __fastcall sub_1623BA0(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 *v5; // r13

  result = **(_QWORD **)(a1 + 56);
  v5 = (__int64 *)(result + 8LL * a2);
  if ( *v5 )
    result = sub_161E7C0((__int64)v5, *v5);
  *v5 = a3;
  if ( a3 )
    return sub_1623A60((__int64)v5, a3, 2);
  return result;
}
