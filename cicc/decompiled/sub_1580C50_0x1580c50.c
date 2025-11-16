// Function: sub_1580C50
// Address: 0x1580c50
//
__int64 __fastcall sub_1580C50(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  result = *(unsigned int *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
