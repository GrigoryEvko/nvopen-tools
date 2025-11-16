// Function: sub_2FDBD80
// Address: 0x2fdbd80
//
bool __fastcall sub_2FDBD80(__int64 a1, __int64 *a2)
{
  bool result; // al
  __int64 v3; // rdx

  result = sub_2E791F0(a2);
  if ( result )
  {
    v3 = *(_QWORD *)(a2[1] + 656);
    if ( *(_DWORD *)(v3 + 336) == 4 )
      return *(_DWORD *)(v3 + 344) == 6 || *(_DWORD *)(v3 + 344) == 0;
  }
  return result;
}
