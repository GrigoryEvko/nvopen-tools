// Function: sub_728750
// Address: 0x728750
//
__int64 __fastcall sub_728750(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  bool v3; // dl
  __int64 v4; // rdi

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
  {
    switch ( (_BYTE)result )
    {
      case 3:
        result = sub_70FCC0(*(_QWORD *)(a1 + 56));
        v3 = (_DWORD)result != 0;
        break;
      case 0x14:
        result = sub_70FCD0(*(_QWORD *)(a1 + 56));
        v3 = (_DWORD)result != 0;
        break;
      case 1:
        result = *(unsigned __int8 *)(a1 + 56);
        if ( (unsigned __int8)(result - 94) <= 1u )
        {
          result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 56LL);
          if ( !*(_QWORD *)(result + 128) )
            return result;
        }
        else
        {
          if ( (_BYTE)result != 4 )
            return result;
          result = (__int64)&dword_4F06970;
          if ( !dword_4F06970 )
            return result;
        }
        goto LABEL_9;
      default:
        return result;
    }
  }
  else
  {
    if ( (_BYTE)result != 2 )
    {
      if ( (_BYTE)result == 3 )
      {
        result = qword_4F04C50;
        if ( !qword_4F04C50 || *(_QWORD *)(qword_4F04C50 + 64LL) != *(_QWORD *)(a1 + 56) )
          return result;
      }
      else if ( (_BYTE)result != 16 )
      {
        return result;
      }
LABEL_9:
      if ( *(_DWORD *)(a2 + 72) )
        return result;
      result = 1;
LABEL_14:
      *(_DWORD *)(a2 + 80) = result;
      *(_DWORD *)(a2 + 72) = 1;
      return result;
    }
    v4 = *(_QWORD *)(a1 + 56);
    result = *(unsigned __int8 *)(v4 + 173);
    if ( (_BYTE)result == 1 )
    {
      v3 = (unsigned int)sub_6210B0(v4, 0) != 0;
      result = v3;
    }
    else
    {
      if ( (_BYTE)result != 6 )
        return result;
      result = sub_70FCE0(v4);
      v3 = (_DWORD)result != 0;
    }
  }
  if ( !*(_DWORD *)(a2 + 72) && v3 )
    goto LABEL_14;
  return result;
}
