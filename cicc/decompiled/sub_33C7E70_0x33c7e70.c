// Function: sub_33C7E70
// Address: 0x33c7e70
//
__int64 __fastcall sub_33C7E70(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 768) = *(_QWORD *)(a1 + 8);
  return result;
}
