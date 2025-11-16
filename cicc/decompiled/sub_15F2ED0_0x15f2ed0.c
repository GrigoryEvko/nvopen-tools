// Function: sub_15F2ED0
// Address: 0x15f2ed0
//
__int64 __fastcall sub_15F2ED0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rbp
  int v3; // eax
  __int64 result; // rax
  char v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // r8
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rdx
  unsigned __int16 v16; // ax
  _QWORD v17[4]; // [rsp-20h] [rbp-20h] BYREF

  v3 = *(unsigned __int8 *)(a1 + 16);
  v17[3] = v2;
  v17[1] = v1;
  switch ( v3 )
  {
    case 29:
      v11 = sub_1560260((_QWORD *)(a1 + 56), -1, 36);
      result = 0;
      if ( !v11 )
      {
        if ( *(char *)(a1 + 23) < 0 )
        {
          v12 = sub_1648A40(a1);
          v14 = v12 + v13;
          v15 = 0;
          if ( *(char *)(a1 + 23) < 0 )
            v15 = sub_1648A40(a1);
          if ( (unsigned int)((v14 - v15) >> 4) )
            goto LABEL_10;
        }
        v10 = *(_QWORD *)(a1 - 72);
        if ( *(_BYTE *)(v10 + 16) )
          goto LABEL_10;
        goto LABEL_17;
      }
      break;
    case 30:
    case 31:
    case 32:
    case 34:
    case 35:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 56:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 71:
    case 72:
    case 73:
    case 75:
    case 76:
    case 77:
    case 79:
    case 80:
    case 81:
      result = 0;
      break;
    case 33:
    case 54:
    case 57:
    case 58:
    case 59:
    case 74:
    case 82:
      goto LABEL_10;
    case 55:
      v16 = *(_WORD *)(a1 + 18);
      if ( ((v16 >> 7) & 6) != 0 )
        LOBYTE(v16) = 1;
      result = v16 & 1;
      break;
    case 78:
      v5 = sub_1560260((_QWORD *)(a1 + 56), -1, 36);
      result = 0;
      if ( !v5 )
      {
        if ( *(char *)(a1 + 23) < 0 )
        {
          v6 = sub_1648A40(a1);
          v8 = v6 + v7;
          v9 = 0;
          if ( *(char *)(a1 + 23) < 0 )
            v9 = sub_1648A40(a1);
          if ( (unsigned int)((v8 - v9) >> 4) )
            goto LABEL_10;
        }
        v10 = *(_QWORD *)(a1 - 24);
        if ( *(_BYTE *)(v10 + 16) )
        {
LABEL_10:
          result = 1;
        }
        else
        {
LABEL_17:
          v17[0] = *(_QWORD *)(v10 + 112);
          result = (unsigned int)sub_1560260(v17, -1, 36) ^ 1;
        }
      }
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
