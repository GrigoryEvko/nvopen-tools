// Function: sub_6DEC70
// Address: 0x6dec70
//
__int64 __fastcall sub_6DEC70(__int64 a1)
{
  char v1; // al
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 result; // rax

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 24);
    if ( v1 == 3 )
      return 1;
    if ( v1 != 1 )
      return 0;
    v2 = *(_BYTE *)(a1 + 56);
    v3 = *(_QWORD *)(a1 + 72);
    if ( v2 == 92 )
      return 1;
    if ( v2 > 0x5Cu )
      break;
    if ( v2 == 3 )
      return 1;
    if ( v2 != 91 )
      return 0;
    a1 = *(_QWORD *)(v3 + 16);
  }
  if ( v2 <= 0x61u )
    return v2 != 93;
  if ( (unsigned __int8)(v2 - 103) > 1u )
    return 0;
  v4 = *(_QWORD *)(v3 + 16);
  result = sub_6DEC70(v4);
  if ( (_DWORD)result )
    return (unsigned int)sub_6DEC70(*(_QWORD *)(v4 + 16)) != 0;
  return result;
}
