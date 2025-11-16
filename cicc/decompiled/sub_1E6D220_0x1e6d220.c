// Function: sub_1E6D220
// Address: 0x1e6d220
//
__int64 __fastcall sub_1E6D220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 512);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 516) )
  {
    sub_16CD150(a1 + 504, (const void *)(a1 + 520), 0, 8, a5, a6);
    result = *(unsigned int *)(a1 + 512);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 504) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 512);
  return result;
}
