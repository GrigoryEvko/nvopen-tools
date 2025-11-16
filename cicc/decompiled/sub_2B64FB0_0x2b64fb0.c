// Function: sub_2B64FB0
// Address: 0x2b64fb0
//
char __fastcall sub_2B64FB0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  _BYTE *v4; // r13
  __int16 v5; // r12
  __int64 *v6; // r14
  char v7; // r8
  char result; // al
  unsigned int v9; // ebx

  v2 = **a1;
  v3 = *(_QWORD *)(v2 + 416);
  v4 = *(_BYTE **)(v2 + 424);
  if ( (unsigned __int8)(*(_BYTE *)v3 - 82) > 1u )
    return *(_BYTE *)a2 == *v4;
  v5 = *(_WORD *)(v3 + 2);
  v6 = (__int64 *)a1[1][413];
  v7 = sub_2B64E30(v3, a2, v6);
  result = 0;
  if ( !v7 )
  {
    result = sub_2B64E30((__int64)v4, a2, v6);
    if ( !result )
    {
      v9 = *(_WORD *)(a2 + 2) & 0x3F;
      return (v5 & 0x3F) != (unsigned int)sub_B52F50(v9) && (v5 & 0x3F) != v9;
    }
  }
  return result;
}
