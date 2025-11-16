// Function: sub_7E4C10
// Address: 0x7e4c10
//
__int64 __fastcall sub_7E4C10(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 240) = result;
  return result;
}
