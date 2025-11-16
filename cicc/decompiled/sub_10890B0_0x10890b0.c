// Function: sub_10890B0
// Address: 0x10890b0
//
__int64 *__fastcall sub_10890B0(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rdi

  result = sub_1088F40(*(_QWORD *)(a1 + 112), a2);
  v3 = *(_QWORD *)(a1 + 120);
  if ( v3 )
    return sub_1088F40(v3, a2);
  return result;
}
