// Function: sub_1210230
// Address: 0x1210230
//
__int64 __fastcall sub_1210230(__int64 a1, _DWORD *a2, int a3)
{
  __int64 v4; // rdi
  unsigned int v5; // eax
  const char *v7; // rax
  unsigned __int64 v8; // rsi
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+20h] [rbp-20h]
  char v11; // [rsp+21h] [rbp-1Fh]

  v4 = a1 + 176;
  v5 = *(_DWORD *)(v4 + 64);
  if ( a3 != 54 )
  {
    switch ( v5 )
    {
      case 0x12Cu:
        *a2 = 32;
        goto LABEL_9;
      case 0x12Du:
        *a2 = 33;
        goto LABEL_9;
      case 0x12Eu:
        *a2 = 40;
        goto LABEL_9;
      case 0x12Fu:
        *a2 = 38;
        goto LABEL_9;
      case 0x130u:
        *a2 = 41;
        goto LABEL_9;
      case 0x131u:
        *a2 = 39;
        goto LABEL_9;
      case 0x132u:
        *a2 = 36;
        goto LABEL_9;
      case 0x133u:
        *a2 = 34;
        goto LABEL_9;
      case 0x134u:
        *a2 = 37;
        goto LABEL_9;
      case 0x135u:
        *a2 = 35;
        goto LABEL_9;
      default:
        v11 = 1;
        v7 = "expected icmp predicate (e.g. 'eq')";
        goto LABEL_21;
    }
  }
  if ( v5 <= 0x13F )
  {
    if ( v5 > 0x131 )
    {
      switch ( v5 )
      {
        case 0x133u:
          *a2 = 10;
          break;
        case 0x134u:
          *a2 = 13;
          break;
        case 0x135u:
          *a2 = 11;
          break;
        case 0x136u:
          *a2 = 1;
          break;
        case 0x137u:
          *a2 = 6;
          break;
        case 0x138u:
          *a2 = 4;
          break;
        case 0x139u:
          *a2 = 2;
          break;
        case 0x13Au:
          *a2 = 5;
          break;
        case 0x13Bu:
          *a2 = 3;
          break;
        case 0x13Cu:
          *a2 = 7;
          break;
        case 0x13Du:
          *a2 = 8;
          break;
        case 0x13Eu:
          *a2 = 9;
          break;
        case 0x13Fu:
          *a2 = 14;
          break;
        default:
          *a2 = 12;
          break;
      }
      goto LABEL_9;
    }
    if ( v5 == 20 )
    {
      *a2 = 15;
      goto LABEL_9;
    }
    if ( v5 == 21 )
    {
      *a2 = 0;
LABEL_9:
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      return 0;
    }
  }
  v11 = 1;
  v7 = "expected fcmp predicate (e.g. 'oeq')";
LABEL_21:
  v8 = *(_QWORD *)(a1 + 232);
  v9 = v7;
  v10 = 3;
  sub_11FD800(v4, v8, (__int64)&v9, 1);
  return 1;
}
