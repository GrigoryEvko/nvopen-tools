// Function: sub_2FF1680
// Address: 0x2ff1680
//
__int64 __fastcall sub_2FF1680(__int64 a1)
{
  _QWORD *v1; // rsi
  _QWORD *v2; // rax
  _QWORD *v3; // rsi
  __int64 result; // rax
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax

  if ( !*(_BYTE *)(a1 + 273) )
  {
    v11 = (_QWORD *)sub_BE0980(1);
    sub_2FF0E80(a1, v11, 0);
  }
  if ( (unsigned int)sub_2FF0570(a1) )
  {
    v7 = (_QWORD *)sub_E00710();
    sub_2FF0E80(a1, v7, 1u);
    v8 = (_QWORD *)sub_DF51B0();
    sub_2FF0E80(a1, v8, 1u);
    v9 = (_QWORD *)sub_D05620();
    sub_2FF0E80(a1, v9, 1u);
    if ( !byte_50292C8 )
    {
      v18 = (_QWORD *)sub_29A76F0();
      sub_2FF0E80(a1, v18, 0);
      v19 = (_QWORD *)sub_285D3E0();
      sub_2FF0E80(a1, v19, 0);
      if ( *(_BYTE *)(a1 + 277) )
      {
        v20 = (_QWORD *)sub_287C030();
        sub_2FF0E80(a1, v20, 0);
      }
    }
    if ( !(_BYTE)qword_5028CA8 )
    {
      v17 = (_QWORD *)sub_28BB450();
      sub_2FF0E80(a1, v17, 0);
    }
    v10 = (_QWORD *)sub_2DC2F80();
    sub_2FF0E80(a1, v10, 0);
  }
  sub_2FF12A0(a1, &unk_501DA25, 0);
  sub_2FF12A0(a1, &unk_5025C0C, 0);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 256) + 564LL) == 5 && !(_BYTE)qword_5028E68 )
  {
    v5 = (_QWORD *)sub_2A2E410();
    sub_2FF0E80(a1, v5, 0);
  }
  v1 = (_QWORD *)sub_3004FC0();
  sub_2FF0E80(a1, v1, 0);
  if ( (unsigned int)sub_2FF0570(a1) && !(_BYTE)qword_50291E8 )
  {
    v14 = (_QWORD *)sub_2730460();
    sub_2FF0E80(a1, v14, 0);
  }
  if ( (unsigned int)sub_2FF0570(a1) && !byte_5027A88 )
  {
    v15 = (_QWORD *)sub_2F817E0();
    sub_2FF0E80(a1, v15, 0);
  }
  if ( (unsigned int)sub_2FF0570(a1) && !(_BYTE)qword_5028F48 )
  {
    v16 = (_QWORD *)sub_28E4630();
    sub_2FF0E80(a1, v16, 0);
  }
  v2 = (_QWORD *)sub_29CE7D0();
  sub_2FF0E80(a1, v2, 0);
  v3 = (_QWORD *)sub_2946730();
  sub_2FF0E80(a1, v3, 0);
  if ( !(_BYTE)qword_50274C8 )
  {
    v6 = (_QWORD *)sub_2DC8EA0();
    sub_2FF0E80(a1, v6, 0);
  }
  result = sub_2FF0570(a1);
  if ( (_DWORD)result && !(_BYTE)qword_50273E8 )
  {
    v13 = (_QWORD *)sub_2F9C2C0();
    result = sub_2FF0E80(a1, v13, 0);
  }
  if ( (_BYTE)qword_50286C8 )
  {
    v12 = (_QWORD *)sub_2DDD300();
    return sub_2FF0E80(a1, v12, 0);
  }
  return result;
}
