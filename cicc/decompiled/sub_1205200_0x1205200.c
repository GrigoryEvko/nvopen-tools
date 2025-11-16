// Function: sub_1205200
// Address: 0x1205200
//
__int64 __fastcall sub_1205200(__int64 a1)
{
  int v1; // ebx
  int v2; // eax
  __int64 result; // rax
  unsigned __int8 *v4; // rbx
  unsigned __int8 *v5; // rax
  __int64 v6; // rcx

  while ( 2 )
  {
    *(_QWORD *)(a1 + 56) = *(_QWORD *)a1;
    v1 = sub_11FD3B0((unsigned __int8 **)a1);
    switch ( v1 )
    {
      case -1:
        return 0;
      case 0:
      case 9:
      case 10:
      case 13:
      case 32:
        continue;
      case 33:
        return sub_11FD500((unsigned __int8 **)a1);
      case 34:
        return sub_11FF280(a1);
      case 35:
        return sub_11FF120((unsigned __int8 **)a1);
      case 36:
        return sub_11FF310(a1);
      case 37:
        return sub_11FF270((unsigned __int8 **)a1);
      case 40:
        return 12;
      case 41:
        return 13;
      case 42:
        return 5;
      case 43:
        return sub_11FD5F0((void **)a1);
      case 44:
        return 4;
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
        return sub_11FEC70(a1);
      case 46:
        v4 = *(unsigned __int8 **)a1;
        v5 = sub_11FD0C0(*(unsigned __int8 **)a1);
        if ( v5 )
        {
          v6 = *(_QWORD *)(a1 + 56);
          *(_QWORD *)a1 = v5;
          sub_2241130(a1 + 72, 0, *(_QWORD *)(a1 + 80), v6, &v5[-v6 - 1]);
          result = 507;
        }
        else if ( *v4 == 46 && v4[1] == 46 )
        {
          *(_QWORD *)a1 = v4 + 2;
          result = 2;
        }
        else
        {
LABEL_4:
          result = 1;
        }
        break;
      case 47:
        if ( (unsigned int)sub_11FD3B0((unsigned __int8 **)a1) == 42 && !(unsigned __int8)sub_11FF460(a1) )
          continue;
        goto LABEL_4;
      case 58:
        return 16;
      case 59:
        sub_11FD3E0((unsigned __int8 **)a1);
        continue;
      case 60:
        return 10;
      case 61:
        return 3;
      case 62:
        return 11;
      case 64:
        return sub_11FF260((unsigned __int8 **)a1);
      case 91:
        return 6;
      case 93:
        return 7;
      case 94:
        return sub_11FF110((unsigned __int8 **)a1);
      case 123:
        return 8;
      case 124:
        return 15;
      case 125:
        return 9;
      default:
        v2 = isalpha((unsigned __int8)v1);
        if ( v1 != 95 && !v2 )
          goto LABEL_4;
        result = sub_11FF4E0(a1);
        break;
    }
    return result;
  }
}
