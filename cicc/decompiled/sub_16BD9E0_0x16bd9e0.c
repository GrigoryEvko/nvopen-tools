// Function: sub_16BD9E0
// Address: 0x16bd9e0
//
__int64 __fastcall sub_16BD9E0(__int64 a1)
{
  __int64 result; // rax

  memset(*(void **)(a1 + 8), 0, 8LL * *(unsigned int *)(a1 + 16));
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(result + 8LL * *(unsigned int *)(a1 + 16)) = -1;
  *(_DWORD *)(a1 + 20) = 0;
  return result;
}
