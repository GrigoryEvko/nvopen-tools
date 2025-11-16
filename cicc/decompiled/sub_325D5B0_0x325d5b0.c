// Function: sub_325D5B0
// Address: 0x325d5b0
//
__int64 __fastcall sub_325D5B0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(result + 768) = *(_QWORD *)(a1 + 8);
  return result;
}
