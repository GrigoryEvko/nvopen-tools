// Function: sub_38E2A90
// Address: 0x38e2a90
//
__int64 __fastcall sub_38E2A90(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 840);
  if ( (_DWORD)result == -1 )
    return *(unsigned int *)(*(_QWORD *)(a1 + 336) + 168LL);
  return result;
}
