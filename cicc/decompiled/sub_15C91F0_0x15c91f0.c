// Function: sub_15C91F0
// Address: 0x15c91f0
//
__int64 __fastcall sub_15C91F0(__int64 a1, _QWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v4; // r8
  __int64 result; // rax

  v4 = *(_QWORD *)(a1 + 24);
  a2[1] = *(_QWORD *)(a1 + 32);
  *a2 = v4;
  *a3 = *(_DWORD *)(a1 + 40);
  result = *(unsigned int *)(a1 + 44);
  *a4 = result;
  return result;
}
