// Function: sub_7304E0
// Address: 0x7304e0
//
__int64 __fastcall sub_7304E0(__int64 a1)
{
  __int64 result; // rax

  while ( 1 )
  {
    result = *(unsigned __int8 *)(a1 + 24);
    *(_BYTE *)(a1 + 25) |= 4u;
    if ( (_BYTE)result != 1 )
      break;
    while ( !(unsigned int)sub_8D2600(*(_QWORD *)a1) )
    {
      result = *(unsigned __int8 *)(a1 + 24);
      if ( (_BYTE)result != 10 )
        return result;
LABEL_3:
      a1 = *(_QWORD *)(a1 + 56);
      result = *(unsigned __int8 *)(a1 + 24);
      *(_BYTE *)(a1 + 25) |= 4u;
      if ( (_BYTE)result != 1 )
        goto LABEL_2;
    }
    result = *(unsigned __int8 *)(a1 + 56);
    a1 = *(_QWORD *)(a1 + 72);
    if ( (_BYTE)result == 91 )
    {
      a1 = *(_QWORD *)(a1 + 16);
    }
    else if ( (unsigned __int8)(result - 103) <= 1u )
    {
      sub_7304E0(*(_QWORD *)(a1 + 16));
      a1 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
    }
    else if ( (_BYTE)result != 5 && (_BYTE)result != 25 )
    {
      return result;
    }
  }
LABEL_2:
  if ( (_BYTE)result == 10 )
    goto LABEL_3;
  return result;
}
