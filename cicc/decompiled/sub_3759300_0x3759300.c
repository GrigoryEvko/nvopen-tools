// Function: sub_3759300
// Address: 0x3759300
//
__int64 __fastcall sub_3759300(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 768) = *(_QWORD *)(a1 + 8);
  return result;
}
