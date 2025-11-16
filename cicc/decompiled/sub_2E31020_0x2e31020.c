// Function: sub_2E31020
// Address: 0x2e31020
//
__int64 __fastcall sub_2E31020(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 96LL);
  *(_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 24)) = 0;
  *(_DWORD *)(a2 + 24) = -1;
  return result;
}
