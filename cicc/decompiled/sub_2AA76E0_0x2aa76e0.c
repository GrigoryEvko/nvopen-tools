// Function: sub_2AA76E0
// Address: 0x2aa76e0
//
bool __fastcall sub_2AA76E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  bool result; // al

  v2 = *(_QWORD *)(a2 + 40);
  result = 0;
  if ( v2 )
    return *(_QWORD *)(v2 + 48) == 0;
  return result;
}
