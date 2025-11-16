// Function: sub_CA57B0
// Address: 0xca57b0
//
char __fastcall sub_CA57B0(__int64 a1, __int64 a2, int a3, int a4)
{
  char result; // al
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 (__fastcall *v9)(__int64, __int64, __int64, _QWORD); // rax

  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 8) = a4;
  result = sub_CA5770(a1);
  if ( result )
  {
    switch ( a3 )
    {
      case 0:
        v7 = 0;
        v8 = 3;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 1:
        v7 = 0;
        v8 = 2;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 2:
        v7 = 0;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_7;
      case 3:
        v7 = 0;
        v8 = 6;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 4:
        v7 = 0;
        v8 = 5;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 5:
        v7 = 0;
        v8 = 1;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 6:
        v7 = 1;
        v8 = 1;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 7:
        v7 = 1;
        v8 = 5;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 8:
        v7 = 1;
        v8 = 0;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
        goto LABEL_5;
      case 9:
        v7 = 1;
        v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2 + 24LL);
LABEL_7:
        v8 = 4;
LABEL_5:
        result = v9(a2, v8, v7, 0);
        break;
      default:
        return result;
    }
  }
  return result;
}
