// Function: sub_D98220
// Address: 0xd98220
//
__int64 __fastcall sub_D98220(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, int *a5)
{
  __int64 result; // rax

  result = 0;
  if ( *(_WORD *)(a2 + 24) == 5 && *(_QWORD *)(a2 + 40) == 2 )
  {
    *a3 = **(_QWORD **)(a2 + 32);
    *a4 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    *a5 = *(_WORD *)(a2 + 28) & 7;
    return 1;
  }
  return result;
}
