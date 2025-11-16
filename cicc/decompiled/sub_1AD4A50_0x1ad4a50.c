// Function: sub_1AD4A50
// Address: 0x1ad4a50
//
__int64 __fastcall sub_1AD4A50(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
