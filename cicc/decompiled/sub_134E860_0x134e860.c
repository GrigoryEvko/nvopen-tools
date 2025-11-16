// Function: sub_134E860
// Address: 0x134e860
//
__int64 __fastcall sub_134E860(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 > 0x17u )
  {
    if ( v1 == 53 )
      return 1;
    goto LABEL_3;
  }
  if ( v1 == 3 )
    return 1;
  if ( v1 > 2u )
  {
LABEL_3:
    result = sub_134E780(a1);
    if ( (_BYTE)result )
      return 1;
    goto LABEL_4;
  }
  if ( v1 != 1 )
    return 1;
  result = sub_134E780(a1);
  if ( (_BYTE)result )
    return 1;
LABEL_4:
  if ( *(_BYTE *)(a1 + 16) != 17 )
    return result;
  if ( !(unsigned __int8)sub_15E04B0(a1) )
    return sub_15E0450(a1);
  return 1;
}
