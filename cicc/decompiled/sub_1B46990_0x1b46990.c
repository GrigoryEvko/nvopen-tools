// Function: sub_1B46990
// Address: 0x1b46990
//
__int64 __fastcall sub_1B46990(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rcx

  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( !v4 )
    return 0xFFFFFFFFLL;
  v5 = 24LL * *(unsigned int *)(a1 + 56) + 8;
  result = 0;
  while ( 1 )
  {
    v7 = a1 - 24LL * v4;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v7 = *(_QWORD *)(a1 - 8);
    if ( *(_QWORD *)(v7 + v5) == a2 )
      break;
    result = (unsigned int)(result + 1);
    v5 += 8;
    if ( v4 == (_DWORD)result )
      return 0xFFFFFFFFLL;
  }
  return result;
}
