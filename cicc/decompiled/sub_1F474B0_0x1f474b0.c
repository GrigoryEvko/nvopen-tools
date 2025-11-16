// Function: sub_1F474B0
// Address: 0x1f474b0
//
__int64 __fastcall sub_1F474B0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  __int64 result; // rax
  _QWORD *v4; // rsi
  _QWORD *v5; // rsi

  switch ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 208) + 608LL) + 348LL) )
  {
    case 0:
      v4 = (_QWORD *)sub_1B22480();
      result = sub_1F46490(a1, v4, 1, 1, 1u);
      if ( !byte_4FCE060 )
      {
        v2 = (_QWORD *)sub_1F56B10();
        goto LABEL_5;
      }
      return result;
    case 1:
    case 3:
      goto LABEL_4;
    case 2:
      v1 = (_QWORD *)sub_2116830();
      goto LABEL_3;
    case 4:
      v1 = (_QWORD *)sub_1F60710(0);
LABEL_3:
      sub_1F46490(a1, v1, 1, 1, 0);
LABEL_4:
      v2 = (_QWORD *)sub_1D82000();
      goto LABEL_5;
    case 5:
      v5 = (_QWORD *)sub_1F60710(0);
      sub_1F46490(a1, v5, 1, 1, 0);
      v2 = (_QWORD *)sub_1F5DD40();
LABEL_5:
      result = sub_1F46490(a1, v2, 1, 1, 0);
      break;
    default:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 208) + 608LL);
      break;
  }
  return result;
}
