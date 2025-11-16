// Function: sub_80A340
// Address: 0x80a340
//
__int64 __fastcall sub_80A340(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = sub_8DBE70(*(_QWORD *)a1);
  if ( (_DWORD)result )
  {
    a2[20] = 1;
    a2[18] = 1;
    return result;
  }
  result = *(unsigned __int8 *)(a1 + 24);
  if ( (unsigned __int8)(result - 11) <= 1u || (_BYTE)result == 14 )
  {
LABEL_6:
    a2[19] = 1;
    return result;
  }
  if ( (_BYTE)result == 7 )
  {
    result = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)result & 1) == 0 )
      goto LABEL_6;
  }
  else if ( (_BYTE)result == 5 || (_BYTE)result == 8 )
  {
    goto LABEL_6;
  }
  result = sub_730740(a1);
  if ( (_DWORD)result && (*(_BYTE *)(a1 + 27) & 2) == 0 )
    goto LABEL_6;
  if ( *(_BYTE *)(a1 + 24) == 1 )
  {
    result = (unsigned int)*(unsigned __int8 *)(a1 + 56) - 22;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 56) - 22) <= 1u )
      goto LABEL_6;
  }
  return result;
}
