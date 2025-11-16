// Function: sub_131D3F0
// Address: 0x131d3f0
//
__int64 __fastcall sub_131D3F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax

  result = *a4;
  *(_QWORD *)(a3 + *a4) = 0xA7A7A7A7A7A7A7A7LL;
  *a4 += 8;
  return result;
}
