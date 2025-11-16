// Function: sub_9363D0
// Address: 0x9363d0
//
void __fastcall sub_9363D0(_QWORD *a1, unsigned __int64 a2)
{
  bool v2; // zf
  char v3; // bl
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // rax
  __int64 v7; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v8[64]; // [rsp+10h] [rbp-40h] BYREF

  v2 = a1[12] == 0;
  v7 = *(_QWORD *)a2;
  if ( v2 )
  {
    v6 = (_QWORD *)sub_945CA0(a1, byte_3F871B3, 0, 0);
    sub_92FEA0((__int64)a1, v6, 0);
  }
  v3 = *(_BYTE *)(a2 + 40);
  if ( v3 == 8 )
  {
    if ( !(_DWORD)v7 && !WORD2(v7) )
      v7 = *(_QWORD *)(*(_QWORD *)(a1[66] + 80LL) + 8LL);
    sub_92FD10((__int64)a1, (unsigned int *)&v7);
    sub_91CAC0(&v7);
    sub_9313C0((__int64)a1, a2);
  }
  else
  {
    sub_92FD10((__int64)a1, (unsigned int *)&v7);
    sub_91CAC0(&v7);
    switch ( v3 )
    {
      case 0:
      case 25:
        sub_921EA0((__int64)v8, (__int64)a1, *(__int64 **)(a2 + 48), 0, 0, 0);
        break;
      case 1:
        sub_937020(a1, a2);
        break;
      case 2:
        sub_936F80(a1, a2);
        break;
      case 5:
        sub_937180(a1, a2);
        break;
      case 6:
        sub_931270((__int64)a1, a2);
        break;
      case 7:
        sub_930570((__int64)a1);
        break;
      case 11:
        sub_9365F0((unsigned int)v8, (_DWORD)a1, a2, 0, 0, 0, 0);
        break;
      case 12:
        sub_936B50(a1, a2);
        break;
      case 13:
        sub_936D30(a1, a2);
        break;
      case 15:
        sub_935670((__int64)a1, a2);
        break;
      case 16:
        sub_9359B0((__int64)a1, (_QWORD *)a2);
        break;
      case 17:
        sub_9303A0(a1, a2);
        break;
      case 18:
        sub_932270((__int64)a1, a2, v4, v5);
        break;
      case 20:
        sub_931670((__int64)a1, a2);
        break;
      case 24:
        return;
      default:
        sub_91B8A0("unsupported statement type", (_DWORD *)a2, 1);
    }
  }
}
