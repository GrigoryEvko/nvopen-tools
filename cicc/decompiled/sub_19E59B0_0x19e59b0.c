// Function: sub_19E59B0
// Address: 0x19e59b0
//
__int64 __fastcall sub_19E59B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx

  result = sub_145CBF0((__int64 *)(a1 + 64), 32, 16);
  *(_DWORD *)(result + 8) = 1;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = &unk_49F4D10;
  v3 = *(unsigned __int8 *)(a2 + 16);
  *(_QWORD *)(result + 24) = a2;
  *(_DWORD *)(result + 12) = v3;
  return result;
}
