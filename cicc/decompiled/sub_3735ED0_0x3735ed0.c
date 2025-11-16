// Function: sub_3735ED0
// Address: 0x3735ed0
//
bool __fastcall sub_3735ED0(__int64 a1, __int64 a2)
{
  bool result; // al
  int v3; // edx

  if ( *(_BYTE *)(a2 + 24) )
    return 0;
  if ( !sub_3247D90(a1, *(_QWORD *)(a2 + 8)) )
  {
    v3 = *(_DWORD *)(a2 + 88);
    result = 1;
    if ( !v3 )
      return result;
    if ( v3 == 1 )
      return sub_3211FB0(*(_QWORD *)(a1 + 208), *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL)) == 0;
  }
  return 0;
}
