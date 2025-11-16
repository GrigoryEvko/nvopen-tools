// Function: sub_302F6C0
// Address: 0x302f6c0
//
__int64 __fastcall sub_302F6C0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax

  v4 = *(_QWORD *)(a3 + 56);
  if ( !v4 )
    return 0;
  v5 = 1;
  while ( 1 )
  {
    while ( a4 != *(_DWORD *)(v4 + 8) )
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
    if ( a4 == *(_DWORD *)(v6 + 8) )
      return 0;
    v4 = *(_QWORD *)(v6 + 32);
    v5 = 0;
    if ( !v4 )
      return v5 ^ 1u;
  }
  return 1;
}
