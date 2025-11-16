// Function: sub_11D9DE0
// Address: 0x11d9de0
//
__int64 __fastcall sub_11D9DE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  while ( a1 )
  {
    v2 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)v2 != 82 || (*(_WORD *)(v2 + 2) & 0x3Fu) - 32 > 1 || *(_QWORD *)(v2 - 32) != a2 )
      return 0;
    a1 = *(_QWORD *)(a1 + 8);
  }
  return 1;
}
