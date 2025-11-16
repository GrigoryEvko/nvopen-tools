// Function: sub_7E4BD0
// Address: 0x7e4bd0
//
__int64 __fastcall sub_7E4BD0(__int64 a1)
{
  __int64 result; // rax

  result = sub_7E4BA0(a1);
  *(_DWORD *)(a1 + 32) &= 0xFFFC07FF;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
