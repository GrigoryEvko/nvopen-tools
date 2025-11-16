// Function: sub_1EA9B60
// Address: 0x1ea9b60
//
__int64 __fastcall sub_1EA9B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  sub_1F03430(a1, a2, a3, a4, a5);
  result = *(_QWORD *)(a1 + 2232);
  if ( result != *(_QWORD *)(a1 + 2240) )
    *(_QWORD *)(a1 + 2240) = result;
  return result;
}
