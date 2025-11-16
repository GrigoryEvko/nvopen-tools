// Function: sub_16F7BE0
// Address: 0x16f7be0
//
__int64 __fastcall sub_16F7BE0(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)result )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 232) + 24LL * (unsigned int)result - 8) == a2 )
    {
      result = (unsigned int)(result - 1);
      *(_DWORD *)(a1 + 240) = result;
    }
  }
  return result;
}
