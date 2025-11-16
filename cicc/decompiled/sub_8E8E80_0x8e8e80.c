// Function: sub_8E8E80
// Address: 0x8e8e80
//
unsigned __int8 *__fastcall sub_8E8E80(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned __int8 *result; // rax

  v2 = *a1;
  if ( *a1 == 69 )
    return a1 + 1;
  if ( v2 == 112 )
  {
    if ( a1[1] != 105 )
    {
LABEL_5:
      result = a1;
      goto LABEL_6;
    }
    result = sub_8E8A30(a1 + 2, 69, 40, 41, 0, a2);
    if ( *result == 69 )
      return ++result;
  }
  else
  {
    if ( v2 != 105 || a1[1] != 108 )
      goto LABEL_5;
    result = sub_8E8A30(a1 + 2, 69, 123, 125, 1, a2);
    if ( *result == 69 )
      return ++result;
  }
LABEL_6:
  if ( !*(_DWORD *)(a2 + 24) )
  {
    ++*(_QWORD *)(a2 + 32);
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
  }
  return result;
}
