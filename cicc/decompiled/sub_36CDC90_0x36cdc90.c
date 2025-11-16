// Function: sub_36CDC90
// Address: 0x36cdc90
//
__int64 __fastcall sub_36CDC90(unsigned __int8 *a1, int a2)
{
  __int64 result; // rax
  unsigned __int8 *v5; // rax

  while ( 1 )
  {
    result = *((_QWORD *)a1 + 1);
    if ( !a2 )
      break;
    if ( *(_BYTE *)(result + 8) == 14 )
    {
      LODWORD(result) = *(_DWORD *)(result + 8) >> 8;
      if ( (_DWORD)result )
        return (unsigned int)result;
    }
    --a2;
    v5 = sub_98ACB0(a1, 1u);
    if ( a1 == v5 )
    {
      result = *((_QWORD *)a1 + 1);
      if ( *(_BYTE *)(result + 8) == 14 )
        goto LABEL_9;
      return 0;
    }
    a1 = v5;
  }
  if ( *(_BYTE *)(result + 8) != 14 )
    return 0;
LABEL_9:
  LODWORD(result) = *(_DWORD *)(result + 8) >> 8;
  return (unsigned int)result;
}
