// Function: sub_1623210
// Address: 0x1623210
//
__int64 __fastcall sub_1623210(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int64 v5; // rdi
  __int64 result; // rax

  v5 = sub_161E760(a2);
  result = 0;
  if ( v5 )
  {
    sub_1623100(v5, a1, a3);
    return 1;
  }
  return result;
}
