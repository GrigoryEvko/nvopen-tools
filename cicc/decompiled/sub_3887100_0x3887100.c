// Function: sub_3887100
// Address: 0x3887100
//
__int64 __fastcall sub_3887100(__int64 a1)
{
  int v1; // ebx
  __int64 result; // rax
  unsigned __int8 *v3; // rbx
  unsigned __int8 *v4; // rax
  _BYTE *v5; // rcx

  while ( 2 )
  {
    *(_QWORD *)(a1 + 48) = *(_QWORD *)a1;
    v1 = sub_3880F40((unsigned __int8 **)a1);
    switch ( v1 )
    {
      case -1:
        result = 0;
        break;
      case 0:
      case 9:
      case 10:
      case 13:
      case 32:
        continue;
      case 33:
        result = sub_3881090((unsigned __int64 *)a1);
        break;
      case 34:
        result = sub_3886ED0(a1);
        break;
      case 35:
        result = sub_3885DD0((unsigned __int8 **)a1);
        break;
      case 36:
        result = sub_3886F80(a1);
        break;
      case 37:
        result = sub_3886EC0((unsigned __int8 **)a1);
        break;
      case 40:
        result = 12;
        break;
      case 41:
        result = 13;
        break;
      case 42:
        result = 5;
        break;
      case 43:
        result = sub_3881180((unsigned __int8 **)a1);
        break;
      case 44:
        result = 4;
        break;
      case 45:
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 56:
      case 57:
        result = sub_3886990((unsigned __int8 **)a1);
        break;
      case 46:
        v3 = *(unsigned __int8 **)a1;
        v4 = sub_3880AC0(*(unsigned __int8 **)a1);
        if ( v4 )
        {
          v5 = *(_BYTE **)(a1 + 48);
          *(_QWORD *)a1 = v4;
          sub_2241130((unsigned __int64 *)(a1 + 64), 0, *(_QWORD *)(a1 + 72), v5, v4 - 1 - v5);
          result = 372;
        }
        else
        {
          result = 1;
          if ( *v3 == 46 && v3[1] == 46 )
          {
            *(_QWORD *)a1 = v3 + 2;
            result = 2;
          }
        }
        break;
      case 58:
        result = 16;
        break;
      case 59:
        sub_3880F70((unsigned __int8 **)a1);
        continue;
      case 60:
        result = 10;
        break;
      case 61:
        result = 3;
        break;
      case 62:
        result = 11;
        break;
      case 64:
        result = sub_3886EB0((unsigned __int8 **)a1);
        break;
      case 91:
        result = 6;
        break;
      case 93:
        result = 7;
        break;
      case 94:
        result = sub_3885DC0((unsigned __int8 **)a1);
        break;
      case 123:
        result = 8;
        break;
      case 124:
        result = 15;
        break;
      case 125:
        result = 9;
        break;
      default:
        if ( isalpha((unsigned __int8)v1) || (result = 1, v1 == 95) )
          result = sub_3881FF0(a1);
        break;
    }
    break;
  }
  return result;
}
