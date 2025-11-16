// Function: sub_2EC8EF0
// Address: 0x2ec8ef0
//
__int64 __fastcall sub_2EC8EF0(__int64 a1, unsigned int a2, int a3)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4LL * a2) += a3;
  result = *(unsigned int *)(*(_QWORD *)(a1 + 192) + 4LL * a2);
  if ( (unsigned int)result > *(_DWORD *)(a1 + 272) )
    *(_DWORD *)(a1 + 272) = result;
  return result;
}
