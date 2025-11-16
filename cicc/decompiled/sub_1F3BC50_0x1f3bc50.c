// Function: sub_1F3BC50
// Address: 0x1f3bc50
//
__int64 __fastcall sub_1F3BC50(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 *v4; // rdx
  __int16 v5; // ax
  __int64 result; // rax
  __int16 v7; // ax
  __int64 v8; // rax
  __int64 (*v9)(); // rax

  v4 = *(__int16 **)(a3 + 16);
  v5 = *v4;
  switch ( *v4 )
  {
    case 0:
    case 8:
    case 10:
    case 14:
    case 15:
    case 45:
LABEL_2:
      result = 0;
      break;
    default:
      switch ( v5 )
      {
        case 2:
        case 3:
        case 4:
        case 6:
        case 9:
        case 12:
        case 13:
        case 17:
        case 18:
          goto LABEL_2;
        default:
          if ( v5 == 1 && (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 8) != 0
            || ((v7 = *(_WORD *)(a3 + 46), (v7 & 4) == 0) && (v7 & 8) != 0
              ? (LOBYTE(v8) = sub_1E15D00(a3, 0x10000u, 1))
              : (v8 = (*((_QWORD *)v4 + 1) >> 16) & 1LL),
                (_BYTE)v8) )
          {
            result = *(unsigned int *)(a2 + 12);
          }
          else
          {
            v9 = *(__int64 (**)())(*(_QWORD *)a1 + 872LL);
            if ( v9 == sub_1D0B180
              || !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v9)(a1, **(unsigned __int16 **)(a3 + 16)) )
            {
              result = 1;
            }
            else
            {
              result = *(unsigned int *)(a2 + 16);
            }
          }
          break;
      }
      break;
  }
  return result;
}
