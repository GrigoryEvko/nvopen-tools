// Function: sub_3189D20
// Address: 0x3189d20
//
_QWORD *__fastcall sub_3189D20(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rbx
  int v3; // r14d
  _QWORD *v4; // r12
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_22077B0(0x28u);
  if ( v2 )
  {
    switch ( *a2 )
    {
      case '*':
        v3 = 27;
        break;
      case '+':
        v3 = 28;
        break;
      case ',':
        v3 = 29;
        break;
      case '-':
        v3 = 30;
        break;
      case '.':
        v3 = 31;
        break;
      case '/':
        v3 = 32;
        break;
      case '0':
        v3 = 33;
        break;
      case '1':
        v3 = 34;
        break;
      case '2':
        v3 = 35;
        break;
      case '3':
        v3 = 36;
        break;
      case '4':
        v3 = 37;
        break;
      case '5':
        v3 = 38;
        break;
      case '6':
        v3 = 39;
        break;
      case '7':
        v3 = 40;
        break;
      case '8':
        v3 = 41;
        break;
      case '9':
        v3 = 42;
        break;
      case ':':
        v3 = 43;
        break;
      case ';':
        v3 = 44;
        break;
      default:
        BUG();
    }
    sub_318EB10(v2, 51, a2, a1);
    *(_DWORD *)(v2 + 32) = v3;
    *(_QWORD *)v2 = &unk_4A34240;
  }
  v6[0] = v2;
  v4 = sub_3189570(a1, (__int64)v6);
  if ( v6[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  return v4;
}
