// Function: sub_732D90
// Address: 0x732d90
//
__int64 __fastcall sub_732D90(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = result;
  *(_QWORD *)(a2 + 24) = a1;
  *(_QWORD *)(a1 + 24) = a2;
  return result;
}
