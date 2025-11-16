// Function: sub_727BE0
// Address: 0x727be0
//
__int64 __fastcall sub_727BE0(char a1, _QWORD *a2)
{
  char v2; // al
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // [rsp+8h] [rbp-18h]

  if ( a2 && (v2 = *((_BYTE *)a2 - 8), (v2 & 1) != 0) )
  {
    if ( (v2 & 2) != 0 )
    {
      if ( *a2 )
      {
        v3 = sub_72B7A0(a2);
      }
      else if ( unk_4D03FE8 )
      {
        v3 = *qword_4D03FD0;
      }
      else
      {
        v3 = unk_4D03FF0;
      }
      result = sub_727B10(64, v3);
    }
    else
    {
      result = sub_7279A0(64);
    }
  }
  else
  {
    result = (__int64)sub_7246D0(64);
  }
  *(_QWORD *)result = 0;
  *(_BYTE *)(result + 8) = a1;
  v5 = *(_QWORD *)&dword_4F077C8;
  *(_BYTE *)(result + 9) = 0;
  *(_BYTE *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = v5;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  switch ( a1 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 8:
    case 9:
    case 10:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 25:
    case 27:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 37:
    case 38:
    case 39:
      return result;
    case 21:
      *(_QWORD *)(result + 56) = 0;
      break;
    case 26:
      *(_BYTE *)(result + 56) = 0;
      break;
    case 28:
    case 29:
      v6 = result;
      sub_726C20((_BYTE *)(result + 56));
      result = v6;
      break;
    default:
      sub_721090();
  }
  return result;
}
