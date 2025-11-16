// Function: sub_1E0BA80
// Address: 0x1e0ba80
//
__int64 __fastcall sub_1E0BA80(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 72);
  if ( !result )
  {
    result = sub_145CBF0((__int64 *)(a1 + 120), 32, 16);
    *(_DWORD *)result = a2;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_QWORD *)(a1 + 72) = result;
  }
  return result;
}
