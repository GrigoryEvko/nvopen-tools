// Function: sub_3736140
// Address: 0x3736140
//
bool __fastcall sub_3736140(__int64 a1)
{
  unsigned __int16 v1; // r8
  bool result; // al

  v1 = sub_3220AA0(*(_QWORD *)(a1 + 208));
  result = 0;
  if ( v1 <= 4u )
    return *(_DWORD *)(*(_QWORD *)(a1 + 208) + 6224LL) != 2;
  return result;
}
