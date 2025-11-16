// Function: sub_109FDD0
// Address: 0x109fdd0
//
char __fastcall sub_109FDD0(__int64 a1)
{
  void *v1; // r12
  char result; // al
  char v3; // dl
  char v4; // dl

  v1 = sub_C33340();
  if ( *(void **)a1 == v1 )
    result = sub_C40310(a1);
  else
    result = sub_C33940(a1);
  if ( result )
    return 0;
  if ( v1 == *(void **)a1 )
  {
    v4 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 20LL) & 7;
    if ( v4 != 1 )
      return v4 != 0 && v4 != 3;
  }
  else
  {
    v3 = *(_BYTE *)(a1 + 20) & 7;
    if ( v3 != 1 )
      return v3 != 0 && v3 != 3;
  }
  return result;
}
