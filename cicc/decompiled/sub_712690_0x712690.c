// Function: sub_712690
// Address: 0x712690
//
_BOOL8 __fastcall sub_712690(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  char i; // dl
  _BOOL8 result; // rax
  char v5[4]; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1;
  if ( *(_BYTE *)(a1 + 173) != 12 )
    return sub_712570(a1);
  v2 = *(_QWORD *)(a1 + 128);
  for ( i = *(_BYTE *)(v2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( i == 2 || (result = 0, i == 14) )
  {
    while ( *(_BYTE *)(v1 + 176) == 1 && (unsigned int)sub_72E9D0(v1, v6, v5) )
    {
      v1 = v6[0];
      if ( *(_BYTE *)(v6[0] + 173) != 12 )
        return 1;
    }
    if ( *(_BYTE *)(v1 + 173) == 12 )
      return (unsigned __int8)(*(_BYTE *)(v1 + 176) - 5) > 1u;
    return 1;
  }
  return result;
}
