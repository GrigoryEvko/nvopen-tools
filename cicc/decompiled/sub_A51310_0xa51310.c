// Function: sub_A51310
// Address: 0xa51310
//
__int64 __fastcall sub_A51310(__int64 a1, unsigned __int8 a2)
{
  unsigned __int8 *v2; // rdx
  __int64 result; // rax

  v2 = *(unsigned __int8 **)(a1 + 32);
  result = a1;
  if ( (unsigned __int64)v2 >= *(_QWORD *)(a1 + 24) )
    return sub_CB5D20(a1, a2);
  *(_QWORD *)(a1 + 32) = v2 + 1;
  *v2 = a2;
  return result;
}
