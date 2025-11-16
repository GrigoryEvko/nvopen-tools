// Function: sub_35D8A40
// Address: 0x35d8a40
//
__int64 __fastcall sub_35D8A40(_QWORD *a1, int a2, unsigned int a3)
{
  __int64 result; // rax

  result = *(unsigned int *)(*a1 + 4LL * a3);
  if ( a2 )
    return (unsigned int)(result - *(_DWORD *)(*a1 + 4LL * (unsigned int)(a2 - 1)));
  return result;
}
