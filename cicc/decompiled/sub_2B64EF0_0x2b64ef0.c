// Function: sub_2B64EF0
// Address: 0x2b64ef0
//
char __fastcall sub_2B64EF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r8
  _BYTE *v3; // r13
  __int16 v4; // r12
  __int64 *v5; // r14
  char v6; // r8
  char result; // al
  unsigned int v8; // ebx

  v2 = *(_QWORD *)(*a1 + 416LL);
  v3 = *(_BYTE **)(*a1 + 424LL);
  if ( (unsigned __int8)(*(_BYTE *)v2 - 82) > 1u )
    return *(_BYTE *)a2 == *v3;
  v4 = *(_WORD *)(v2 + 2);
  v5 = *(__int64 **)(a1[1] + 3304LL);
  v6 = sub_2B64E30(*(_QWORD *)(*a1 + 416LL), a2, v5);
  result = 0;
  if ( !v6 )
  {
    result = sub_2B64E30((__int64)v3, a2, v5);
    if ( !result )
    {
      v8 = *(_WORD *)(a2 + 2) & 0x3F;
      return (v4 & 0x3F) != (unsigned int)sub_B52F50(v8) && (v4 & 0x3F) != v8;
    }
  }
  return result;
}
