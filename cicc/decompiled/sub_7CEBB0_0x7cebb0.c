// Function: sub_7CEBB0
// Address: 0x7cebb0
//
_QWORD *__fastcall sub_7CEBB0(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // r12

  while ( 1 )
  {
    result = (_QWORD *)*a2;
    if ( *a2 )
      break;
LABEL_6:
    v4 = (_QWORD *)sub_8787C0();
    v4[1] = a1;
    *v4 = *a2;
    *a2 = v4;
    if ( a1 )
    {
      result = (_QWORD *)sub_85EB10(*(_QWORD *)(a1 + 128));
      v5 = (_QWORD *)result[16];
      if ( !v5 )
        goto LABEL_10;
    }
    else
    {
      result = (_QWORD *)sub_85EB10(unk_4F07288);
      v5 = (_QWORD *)result[16];
      if ( !v5 )
        return result;
    }
    do
    {
      result = (_QWORD *)sub_7CEBB0(v5[1], a2);
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
    if ( !a1 )
      return result;
LABEL_10:
    if ( (*(_BYTE *)(a1 + 124) & 0xA) == 0 )
      return result;
    a1 = *(_QWORD *)(a1 + 40);
    if ( a1 )
    {
      if ( *(_BYTE *)(a1 + 28) == 3 )
        a1 = *(_QWORD *)(a1 + 32);
      else
        a1 = 0;
    }
  }
  while ( result[1] != a1 )
  {
    result = (_QWORD *)*result;
    if ( !result )
      goto LABEL_6;
  }
  return result;
}
