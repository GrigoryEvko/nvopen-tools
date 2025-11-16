// Function: sub_341F2D0
// Address: 0x341f2d0
//
__int64 __fastcall sub_341F2D0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // r8d
  __int64 v4; // rax
  int v5; // ecx
  __int64 v6; // rax

  v3 = 0;
  if ( !*(_DWORD *)(a1 + 792) )
    return v3;
  v4 = *(_QWORD *)(a2 + 56);
  if ( !v4 )
    return v3;
  v5 = 1;
  while ( 1 )
  {
    while ( a3 != *(_DWORD *)(v4 + 8) )
    {
      v4 = *(_QWORD *)(v4 + 32);
      if ( !v4 )
        return v5 ^ 1u;
    }
    if ( !v5 )
      return 0;
    v6 = *(_QWORD *)(v4 + 32);
    if ( !v6 )
      break;
    if ( a3 == *(_DWORD *)(v6 + 8) )
      return 0;
    v4 = *(_QWORD *)(v6 + 32);
    v5 = 0;
    if ( !v4 )
      return v5 ^ 1u;
  }
  return 1;
}
