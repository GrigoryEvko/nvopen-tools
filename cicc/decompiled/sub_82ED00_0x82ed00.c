// Function: sub_82ED00
// Address: 0x82ed00
//
_BOOL8 __fastcall sub_82ED00(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rdx

  if ( (unsigned int)sub_8DBE70(*(_QWORD *)a1) )
    return 1;
  result = sub_82EC00(a1);
  if ( result )
    return 1;
  v3 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v3 == 1 )
    return sub_731E00(*(_QWORD *)(a1 + 144));
  if ( (_BYTE)v3 == 5 )
    return (unsigned int)sub_82EC50(*(_QWORD *)(a1 + 144), a2, v3) != 0;
  return result;
}
