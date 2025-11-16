// Function: sub_6DEEF0
// Address: 0x6deef0
//
__int64 __fastcall sub_6DEEF0(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  _QWORD *v4; // r13

  v2 = a1;
  if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
  {
    result = (unsigned int)a2[25];
    if ( !(_DWORD)result || (result = sub_8D3A70(*(_QWORD *)a1), !(_DWORD)result) )
    {
      if ( *(_BYTE *)(a1 + 24) == 1 && *(_BYTE *)(a1 + 56) == 5 )
      {
        result = sub_8D2E30(*(_QWORD *)a1);
        if ( (_DWORD)result )
        {
          v4 = *(_QWORD **)(a1 + 72);
          result = sub_8D2E30(*v4);
          if ( (_DWORD)result )
          {
            result = sub_76CDC0(v4);
            a2[19] = 1;
          }
        }
      }
      return result;
    }
  }
  while ( 1 )
  {
    result = *(unsigned __int8 *)(v2 + 24);
    if ( (_BYTE)result != 1 )
      break;
    while ( 1 )
    {
      result = *(unsigned __int8 *)(v2 + 56);
      if ( (((_BYTE)result - 6) & 0xFD) != 0 )
        break;
      v2 = *(_QWORD *)(v2 + 72);
      result = *(unsigned __int8 *)(v2 + 24);
      if ( (_BYTE)result != 1 )
        goto LABEL_12;
    }
    if ( (_BYTE)result != 4 )
      return result;
    result = *(_QWORD *)(v2 + 72);
    if ( *(_BYTE *)(result + 24) != 1 || *(_BYTE *)(result + 56) != 1 )
      return result;
    v2 = *(_QWORD *)(result + 72);
  }
LABEL_12:
  switch ( (_BYTE)result )
  {
    case 3:
      result = *(_QWORD *)(v2 + 56);
      if ( *(_BYTE *)(result + 136) <= 2u )
        return result;
LABEL_14:
      a2[20] = 1;
      a2[37] = 0;
      a2[18] = 1;
      return result;
    case 0x18:
      goto LABEL_14;
    case 5:
      result = *(_QWORD *)(v2 + 56);
      if ( (*(_BYTE *)(result + 49) & 1) == 0 )
      {
        a2[20] = 1;
        a2[37] = 1;
        a2[18] = 1;
      }
      break;
  }
  return result;
}
