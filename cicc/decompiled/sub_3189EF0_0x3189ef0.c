// Function: sub_3189EF0
// Address: 0x3189ef0
//
_QWORD *__fastcall sub_3189EF0(__int64 a1, _BYTE *a2)
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
      case 'C':
        v3 = 57;
        break;
      case 'D':
        v3 = 48;
        break;
      case 'E':
        v3 = 49;
        break;
      case 'F':
        v3 = 50;
        break;
      case 'G':
        v3 = 51;
        break;
      case 'H':
        v3 = 56;
        break;
      case 'I':
        v3 = 55;
        break;
      case 'J':
        v3 = 58;
        break;
      case 'K':
        v3 = 52;
        break;
      case 'L':
        v3 = 53;
        break;
      case 'M':
        v3 = 54;
        break;
      case 'N':
        v3 = 59;
        break;
      case 'O':
        v3 = 60;
        break;
      default:
        BUG();
    }
    sub_318EB10(v2, 55, a2, a1);
    *(_DWORD *)(v2 + 32) = v3;
    *(_QWORD *)v2 = &unk_4A34400;
  }
  v6[0] = v2;
  v4 = sub_3189570(a1, (__int64)v6);
  if ( v6[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  return v4;
}
