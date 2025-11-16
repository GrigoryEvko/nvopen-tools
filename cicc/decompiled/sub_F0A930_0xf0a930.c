// Function: sub_F0A930
// Address: 0xf0a930
//
__int64 __fastcall sub_F0A930(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  __int64 v4; // rax

  v2 = *(_QWORD *)(a1 - 8);
  v3 = 0x1FFFFFFFE0LL;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
    return *(_QWORD *)(v2 + v3);
  v4 = 0;
  do
  {
    if ( a2 == *(_QWORD *)(v2 + 32LL * *(unsigned int *)(a1 + 72) + 8 * v4) )
    {
      v3 = 32 * v4;
      return *(_QWORD *)(v2 + v3);
    }
    ++v4;
  }
  while ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != (_DWORD)v4 );
  return *(_QWORD *)(v2 + 0x1FFFFFFFE0LL);
}
