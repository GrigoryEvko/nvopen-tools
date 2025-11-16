// Function: sub_8E3660
// Address: 0x8e3660
//
__int64 __fastcall sub_8E3660(__int64 a1)
{
  char v1; // al
  unsigned __int8 v2; // dl
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 140);
  if ( v1 == 12 )
    return *(_QWORD *)(a1 + 168) + 48LL;
  v2 = v1 - 9;
  result = 0;
  if ( v2 <= 2u )
    return *(_QWORD *)(a1 + 168) + 232LL;
  return result;
}
