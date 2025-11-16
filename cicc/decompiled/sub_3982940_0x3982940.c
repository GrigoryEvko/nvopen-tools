// Function: sub_3982940
// Address: 0x3982940
//
__int64 __fastcall sub_3982940(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdi
  __int16 v4; // dx
  __int64 v5; // rdi

  v4 = *(_WORD *)(a1 + 6);
  switch ( *(_DWORD *)a1 )
  {
    case 0:
    case 0xA:
      result = (unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 1;
      break;
    case 1:
      result = sub_3982020((unsigned __int64 *)(a1 + 8), a2, v4);
      break;
    case 2:
      result = sub_39824E0((__int64 *)(a1 + 8), a2, v4);
      break;
    case 3:
      result = sub_3982260(a1 + 8, a2, v4);
      break;
    case 4:
      result = sub_39822D0(a1 + 8, a2, v4);
      break;
    case 5:
      result = sub_3982400(*(_QWORD *)(a1 + 8), a2, v4);
      break;
    case 6:
      result = sub_3982640(a1 + 8, a2, v4);
      break;
    case 7:
      v5 = *(_QWORD *)(a1 + 8);
      switch ( v4 )
      {
        case 3:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 2);
          break;
        case 4:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 4);
          break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
          result = 16;
          break;
        case 9:
          goto LABEL_4;
        case 10:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 1);
          break;
      }
      break;
    case 8:
      v5 = *(_QWORD *)(a1 + 8);
      switch ( v4 )
      {
        case 3:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 2);
          break;
        case 4:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 4);
          break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
LABEL_4:
          v3 = *(unsigned int *)(v5 + 8);
          result = (unsigned int)v3 + (unsigned int)sub_3946290(v3);
          break;
        case 10:
          result = (unsigned int)(*(_DWORD *)(v5 + 8) + 1);
          break;
      }
      break;
    case 9:
      if ( v4 == 23 || v4 == 6 )
        result = 4;
      else
        result = *(unsigned int *)(*(_QWORD *)(a2 + 240) + 8LL);
      break;
  }
  return result;
}
