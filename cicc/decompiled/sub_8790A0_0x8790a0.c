// Function: sub_8790A0
// Address: 0x8790a0
//
__int64 __fastcall sub_8790A0(__int64 a1)
{
  __int64 result; // rax

  result = sub_823970(24);
  *(_DWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(a1 + 104) = result;
  return result;
}
