// Function: sub_1648CD0
// Address: 0x1648cd0
//
bool __fastcall sub_1648CD0(__int64 a1, int a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( !a2 )
    return v2 == 0;
  while ( v2 )
  {
    v2 = *(_QWORD *)(v2 + 8);
    if ( !--a2 )
      return v2 == 0;
  }
  return 0;
}
