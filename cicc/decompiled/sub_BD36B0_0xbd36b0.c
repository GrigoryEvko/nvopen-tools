// Function: sub_BD36B0
// Address: 0xbd36b0
//
__int64 __fastcall sub_BD36B0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 8);
    if ( !v2 )
      return 1;
    while ( *(_QWORD *)(v2 + 24) == *(_QWORD *)(v1 + 24) )
    {
      v2 = *(_QWORD *)(v2 + 8);
      v1 = *(_QWORD *)(v1 + 8);
      if ( !v2 )
        return 1;
    }
  }
  return 0;
}
