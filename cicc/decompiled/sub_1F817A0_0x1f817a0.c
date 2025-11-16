// Function: sub_1F817A0
// Address: 0x1f817a0
//
__int64 __fastcall sub_1F817A0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 4, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = *a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
