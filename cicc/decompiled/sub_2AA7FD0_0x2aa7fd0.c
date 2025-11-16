// Function: sub_2AA7FD0
// Address: 0x2aa7fd0
//
bool __fastcall sub_2AA7FD0(_QWORD **a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rbx
  __int64 v4; // rdx

  result = 1;
  v3 = *(_QWORD *)(a2 + 40);
  if ( !*(_QWORD *)(v3 + 48) )
  {
    v4 = sub_2BF3F10(**a1);
    result = 0;
    if ( *(_DWORD *)(v4 + 64) == 1 )
      return **(_QWORD **)(v4 + 56) == v3;
  }
  return result;
}
