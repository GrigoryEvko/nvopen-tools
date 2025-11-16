// Function: sub_25FB9B0
// Address: 0x25fb9b0
//
__int64 __fastcall sub_25FB9B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  char v4; // cl
  unsigned __int16 v5; // dx

  v2 = *(_QWORD *)(a2 - 32);
  if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v2 + 24) )
  {
    if ( !sub_B491E0(a2) )
      goto LABEL_9;
  }
  else if ( !sub_B491E0(a2) )
  {
    return 0;
  }
  if ( !*(_BYTE *)(a1 + 1) )
    return 0;
LABEL_9:
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 53) )
    return 0;
  v4 = sub_B49560(a2, 53);
  if ( v4 )
    return 0;
  v5 = *(_WORD *)(a2 + 2);
  if ( ((((v5 >> 2) & 0x3FF) - 18) & 0xFFFD) == 0 )
  {
    v4 = *(_BYTE *)(a1 + 3);
    if ( !v4 )
      return 0;
  }
  result = 1;
  if ( (v5 & 3) == 2 )
    return (unsigned __int8)(v4 & *(_BYTE *)(a1 + 3));
  return result;
}
