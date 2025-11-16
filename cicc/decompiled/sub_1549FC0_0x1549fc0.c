// Function: sub_1549FC0
// Address: 0x1549fc0
//
__int64 __fastcall sub_1549FC0(__int64 a1, unsigned __int8 a2)
{
  unsigned __int8 *v2; // rdx
  __int64 result; // rax

  v2 = *(unsigned __int8 **)(a1 + 24);
  result = a1;
  if ( (unsigned __int64)v2 >= *(_QWORD *)(a1 + 16) )
    return sub_16E7DE0(a1, a2);
  *(_QWORD *)(a1 + 24) = v2 + 1;
  *v2 = a2;
  return result;
}
