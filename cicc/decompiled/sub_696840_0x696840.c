// Function: sub_696840
// Address: 0x696840
//
__int64 __fastcall sub_696840(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r13d
  __int64 v4; // rax

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 1 )
  {
    if ( (unsigned int)sub_731EE0(*(_QWORD *)(a1 + 144)) )
      return 1;
    v1 = *(_BYTE *)(a1 + 16);
  }
  if ( v1 == 2 && (unsigned int)sub_7322D0(a1 + 144) )
    return 1;
  v2 = sub_82EC00(a1);
  if ( v2 )
    return 1;
  if ( *(_BYTE *)(a1 + 17) == 1 && !(unsigned int)sub_6ED0A0(a1) )
  {
    v4 = sub_6ED2B0(a1);
    if ( v4 )
      return *(_BYTE *)(v4 + 173) == 12;
  }
  return v2;
}
