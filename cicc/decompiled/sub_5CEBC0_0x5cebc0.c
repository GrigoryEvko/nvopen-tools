// Function: sub_5CEBC0
// Address: 0x5cebc0
//
_BYTE *__fastcall sub_5CEBC0(_QWORD *a1, _BYTE *a2, char a3)
{
  __int64 v4; // rdx
  _QWORD **v5; // rdx
  _QWORD *i; // rax
  __int64 v7; // rax
  char v8; // al
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( a3 == 11 )
  {
    sub_73EA10(a2 + 152, v9);
    if ( *(_BYTE *)(v9[0] + 140LL) == 7 )
    {
      v4 = a1[4];
      *(_BYTE *)(*(_QWORD *)(v9[0] + 168LL) + 20LL) |= 0x80u;
      if ( *(_BYTE *)(*(_QWORD *)(v4 + 40) + 173LL) != 12 && !(unsigned int)sub_711520() )
      {
        v8 = a2[174];
        if ( v8 == 1 )
        {
          a2[193] |= 0x80u;
        }
        else if ( v8 == 3 )
        {
          a2[194] |= 1u;
        }
      }
      v5 = (_QWORD **)sub_5CEB70((__int64)a2, 11);
      for ( i = *v5; a1 != i; i = (_QWORD *)*i )
        v5 = (_QWORD **)i;
      *v5 = (_QWORD *)*a1;
      v7 = v9[0];
      *a1 = *(_QWORD *)(v9[0] + 104LL);
      *(_QWORD *)(v7 + 104) = a1;
    }
  }
  return a2;
}
