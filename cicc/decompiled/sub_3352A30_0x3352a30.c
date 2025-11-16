// Function: sub_3352A30
// Address: 0x3352a30
//
__int64 __fastcall sub_3352A30(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned int *)(a2 + 200)) = 0;
  result = *(unsigned int *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned int *)(a2 + 200));
  if ( !(_DWORD)result )
    return sub_3352560(a2, (__int64 *)(a1 + 96));
  return result;
}
