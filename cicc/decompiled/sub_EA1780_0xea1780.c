// Function: sub_EA1780
// Address: 0xea1780
//
__int64 __fastcall sub_EA1780(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  void *v3; // rax
  bool v4; // r8

  if ( sub_EA1770(a1) )
  {
    if ( (unsigned __int16)(((*(_WORD *)(a1 + 12) >> 3) & 3) - 1) <= 2u )
      return dword_3F82A70[(unsigned __int16)(((*(_WORD *)(a1 + 12) >> 3) & 3) - 1)];
    return 0;
  }
  if ( *(_QWORD *)a1 )
    return 0;
  v2 = *(_BYTE *)(a1 + 9);
  if ( (v2 & 0x70) == 0x20 && *(char *)(a1 + 8) >= 0 )
  {
    *(_BYTE *)(a1 + 8) |= 8u;
    v3 = sub_E807D0(*(_QWORD *)(a1 + 24));
    *(_QWORD *)a1 = v3;
    if ( !v3 )
    {
      v2 = *(_BYTE *)(a1 + 9);
      goto LABEL_11;
    }
    return 0;
  }
LABEL_11:
  result = 1;
  if ( (v2 & 8) == 0 )
  {
    v4 = sub_EA16D0(a1);
    result = 2;
    if ( !v4 )
      return (unsigned __int8)sub_EA16F0(a1) ^ 1u;
  }
  return result;
}
