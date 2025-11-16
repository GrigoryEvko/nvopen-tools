// Function: sub_341E3A0
// Address: 0x341e3a0
//
__int64 __fastcall sub_341E3A0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 768) = *(_QWORD *)(a1 + 8);
  return result;
}
