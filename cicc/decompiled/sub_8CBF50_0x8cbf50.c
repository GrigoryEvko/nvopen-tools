// Function: sub_8CBF50
// Address: 0x8cbf50
//
unsigned __int64 __fastcall sub_8CBF50(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rcx
  int v4; // eax
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx

  v2 = *(unsigned __int8 *)(a2 + 80);
  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      goto LABEL_3;
    case 6:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      goto LABEL_3;
    case 9:
    case 0xA:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      goto LABEL_3;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v3 = *(_QWORD *)(a1 + 88);
LABEL_3:
      v4 = (unsigned __int8)(v2 - 4);
      switch ( (char)v2 )
      {
        case 4:
        case 5:
          LOBYTE(a1) = v3 != 0;
          goto LABEL_17;
        case 6:
          LOBYTE(a1) = v3 != 0;
          goto LABEL_21;
        case 9:
        case 10:
          LOBYTE(a1) = v3 != 0;
          goto LABEL_5;
        case 19:
        case 20:
        case 21:
        case 22:
          LOBYTE(a1) = v3 != 0;
          goto LABEL_14;
        default:
          result = (unsigned int)(v2 - 4);
          break;
      }
      break;
    default:
      v4 = (unsigned __int8)(v2 - 4);
      switch ( (char)v2 )
      {
        case 4:
        case 5:
          LODWORD(a1) = 0;
          v3 = 0;
LABEL_17:
          v5 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
          LOBYTE(v4) = v5 != 0;
          result = (unsigned int)a1 & v4;
          goto LABEL_6;
        case 6:
          LODWORD(a1) = 0;
          v3 = 0;
LABEL_21:
          v5 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
          LOBYTE(v4) = v5 != 0;
          result = (unsigned int)a1 & v4;
          goto LABEL_6;
        case 9:
        case 10:
          LODWORD(a1) = 0;
          v3 = 0;
LABEL_5:
          v5 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
          LOBYTE(v4) = v5 != 0;
          result = (unsigned int)a1 & v4;
          goto LABEL_6;
        case 19:
        case 20:
        case 21:
        case 22:
          LODWORD(a1) = 0;
          v3 = 0;
LABEL_14:
          v5 = *(_QWORD *)(a2 + 88);
          LOBYTE(v4) = v5 != 0;
          result = (unsigned int)a1 & v4;
LABEL_6:
          if ( (_BYTE)result )
          {
            v7 = *(_QWORD *)(v3 + 104);
            v8 = *(_QWORD *)(v5 + 104);
            if ( v7 )
            {
              if ( v8 )
              {
                result = *(unsigned __int8 *)(v8 + 120);
                if ( *(_BYTE *)(v7 + 120) == (_BYTE)result )
                  result = (unsigned __int64)sub_8CBB20(0x3Bu, v7, (_QWORD *)v8);
              }
            }
          }
          break;
        default:
          result = (unsigned int)(v2 - 4);
          break;
      }
      break;
  }
  return result;
}
