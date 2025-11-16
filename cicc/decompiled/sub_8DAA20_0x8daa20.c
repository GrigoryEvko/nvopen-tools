// Function: sub_8DAA20
// Address: 0x8daa20
//
__int64 __fastcall sub_8DAA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // r12
  char i; // dl
  unsigned int v8; // r8d

  v4 = a1;
  v5 = *(_BYTE *)(a1 + 140);
  v6 = a2;
  if ( v5 != 12 )
    goto LABEL_5;
  do
  {
    v4 = *(_QWORD *)(v4 + 160);
    v5 = *(_BYTE *)(v4 + 140);
  }
  while ( v5 == 12 );
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v6 + 140) )
  {
    v6 = *(_QWORD *)(v6 + 160);
LABEL_5:
    ;
  }
  v8 = 0;
  if ( (unsigned __int8)(v5 - 9) <= 2u && (unsigned __int8)(i - 9) <= 2u )
    return v6 == v4 || (unsigned int)sub_8D97D0(v4, v6, 0, a4, 0) || sub_8D5CE0(v4, v6) || sub_8D5CE0(v6, v4) != 0;
  return v8;
}
