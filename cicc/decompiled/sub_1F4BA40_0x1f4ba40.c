// Function: sub_1F4BA40
// Address: 0x1f4ba40
//
__int64 __fastcall sub_1F4BA40(__int64 a1, __int64 a2, _WORD *a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int16 v6; // ax

  if ( sub_1F4B690(a1) )
  {
    v4 = *(_QWORD *)(a1 + 168);
    if ( v4 )
    {
      result = (unsigned int)*(__int16 *)(v4 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL));
      if ( (int)result < 0 )
        return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 184) + 824LL))(
                 *(_QWORD *)(a1 + 184),
                 a1 + 72,
                 a2);
    }
    else
    {
      return 1;
    }
  }
  else
  {
    if ( !sub_1F4B670(a1) )
      goto LABEL_16;
    if ( !a3 )
      a3 = sub_1F4B8B0(a1, a2);
    LOWORD(result) = *a3 & 0x3FFF;
    if ( (_WORD)result == 0x3FFF )
    {
LABEL_16:
      v6 = **(_WORD **)(a2 + 16);
      switch ( v6 )
      {
        case 0:
        case 8:
        case 10:
        case 14:
        case 15:
        case 45:
LABEL_12:
          result = 0;
          break;
        default:
          switch ( v6 )
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
              goto LABEL_12;
            default:
              return 1;
          }
      }
    }
    else
    {
      return (unsigned __int16)result;
    }
  }
  return result;
}
