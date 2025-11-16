// Function: sub_EA2290
// Address: 0xea2290
//
__int64 __fastcall sub_EA2290(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 864);
  if ( (_DWORD)result == -1 )
    return *(unsigned int *)(*(_QWORD *)(a1 + 240) + 176LL);
  return result;
}
