// Function: sub_1B19130
// Address: 0x1b19130
//
__int64 __fastcall sub_1B19130(__int64 a1, _BYTE *a2, int a3, __int64 a4, char a5)
{
  _BYTE *v8; // rdx
  char v9; // al
  __int64 v11; // rax
  int v12; // edx

  v8 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 8LL) - 1) <= 5u && !v8 )
  {
    v9 = sub_15F2480((__int64)a2);
    v8 = 0;
    if ( !v9 )
      v8 = a2;
  }
  switch ( a2[16] )
  {
    case '#':
    case '%':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 1;
      break;
    case '$':
    case '&':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = v8;
      *(_BYTE *)a1 = a3 == 7;
      break;
    case '\'':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 2;
      break;
    case '(':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = v8;
      *(_BYTE *)a1 = a3 == 8;
      break;
    case '2':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 4;
      break;
    case '3':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 3;
      break;
    case '4':
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_BYTE *)a1 = a3 == 5;
      break;
    case 'K':
    case 'L':
    case 'O':
      if ( a3 != 6 && (a5 != 1 || a3 != 9) )
        goto LABEL_9;
      sub_1B18DD0(a1, (__int64)a2, a4);
      break;
    case 'M':
      v11 = *(_QWORD *)(a4 + 24);
      v12 = *(_DWORD *)(a4 + 16);
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = v12;
      *(_QWORD *)(a1 + 24) = v11;
      break;
    default:
LABEL_9:
      *(_BYTE *)a1 = 0;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      break;
  }
  return a1;
}
