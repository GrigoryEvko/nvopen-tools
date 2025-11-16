// Function: sub_10FDBE0
// Address: 0x10fdbe0
//
__int64 __fastcall sub_10FDBE0(__int64 a1, char a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rax
  _DWORD *v4; // rax
  _DWORD *v5; // rax
  _DWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v9; // rax
  _DWORD *v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax

  v2 = *(_QWORD *)(a1 + 8);
  v3 = (_QWORD *)sub_BD5C60(a1);
  if ( v2 == sub_BCB1C0(v3) )
    return 0;
  if ( a2 )
  {
    v10 = sub_C33300();
    if ( !sub_10FDB00(a1, v10) )
      goto LABEL_4;
    v11 = (_QWORD *)sub_BD5C60(a1);
    return sub_BCB150(v11);
  }
  else
  {
    v4 = sub_C332F0();
    if ( !sub_10FDB00(a1, v4) )
    {
LABEL_4:
      v5 = sub_C33310();
      if ( sub_10FDB00(a1, v5) )
      {
        v9 = (_QWORD *)sub_BD5C60(a1);
        return sub_BCB160(v9);
      }
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 3 )
      {
        v6 = sub_C33320();
        if ( sub_10FDB00(a1, v6) )
        {
          v7 = (_QWORD *)sub_BD5C60(a1);
          return sub_BCB170(v7);
        }
      }
      return 0;
    }
    v12 = (_QWORD *)sub_BD5C60(a1);
    return sub_BCB140(v12);
  }
}
