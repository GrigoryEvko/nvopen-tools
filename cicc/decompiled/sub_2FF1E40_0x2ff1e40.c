// Function: sub_2FF1E40
// Address: 0x2ff1e40
//
__int64 __fastcall sub_2FF1E40(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rax
  int v4; // eax
  _QWORD *v5; // rax
  __int64 result; // rax
  _QWORD *v7; // rsi
  _QWORD *v8; // rsi

  v2 = *(_QWORD *)(a1 + 256);
  switch ( *(_DWORD *)(*(_QWORD *)(v2 + 656) + 336LL) )
  {
    case 0:
      v7 = (_QWORD *)sub_2A309A0();
      sub_2FF0E80(a1, v7, 1u);
      v5 = (_QWORD *)sub_3004FC0();
      goto LABEL_5;
    case 1:
    case 3:
    case 6:
    case 7:
      goto LABEL_4;
    case 2:
      v3 = (_QWORD *)sub_2FA88A0(v2);
      goto LABEL_3;
    case 4:
      v3 = (_QWORD *)sub_3012340(0);
LABEL_3:
      sub_2FF0E80(a1, v3, 0);
LABEL_4:
      v4 = sub_2FF0570(a1);
      v5 = (_QWORD *)sub_2DB0110(v4);
      goto LABEL_5;
    case 5:
      v8 = (_QWORD *)sub_3012340(1);
      sub_2FF0E80(a1, v8, 0);
      v5 = (_QWORD *)sub_300FC50();
LABEL_5:
      result = sub_2FF0E80(a1, v5, 0);
      break;
    default:
      result = *(_QWORD *)(v2 + 656);
      break;
  }
  return result;
}
