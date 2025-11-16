// Function: sub_87DE40
// Address: 0x87de40
//
__int64 __fastcall sub_87DE40(__int64 a1, __int64 a2)
{
  _QWORD *i; // r15
  _QWORD *v3; // r13
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // r12
  char v7; // al
  _QWORD *v8; // r14

  for ( i = *(_QWORD **)(a1 + 112); i; i = (_QWORD *)*i )
  {
    v3 = (_QWORD *)i[2];
    v4 = (_QWORD *)i[1];
    if ( v4 == (_QWORD *)*v3 )
      return 1;
    v5 = a2;
    while ( 1 )
    {
      v6 = v4[2];
      if ( v3 == v4 )
        break;
      v7 = *(_BYTE *)(v6 + 96);
      v8 = *(_QWORD **)(v6 + 112);
      if ( (v7 & 2) == 0 || (v7 & 1) != 0 && !*v8 )
        goto LABEL_12;
      if ( !(unsigned int)sub_87DE40(v4[2], v5) )
        goto LABEL_15;
LABEL_9:
      v5 = *(_QWORD *)(v6 + 40);
      v4 = (_QWORD *)*v4;
      if ( (_QWORD *)*v3 == v4 )
        return 1;
    }
    v8 = i;
LABEL_12:
    if ( !*((_BYTE *)v8 + 25)
      || (unsigned int)sub_87D890(v5)
      || *((_BYTE *)v8 + 25) == 1 && (unsigned int)sub_87D970(v5) )
    {
      goto LABEL_9;
    }
LABEL_15:
    ;
  }
  return 0;
}
