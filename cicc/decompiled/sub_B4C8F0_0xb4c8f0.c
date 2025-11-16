// Function: sub_B4C8F0
// Address: 0xb4c8f0
//
_QWORD *__fastcall sub_B4C8F0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned __int16 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *result; // rax

  v8 = sub_AA48A0(a2);
  v9 = sub_BCB120(v8);
  result = sub_B44260(a1, v9, 2, a3, a4, a5);
  if ( *(_QWORD *)(a1 - 32) )
  {
    result = *(_QWORD **)(a1 - 24);
    **(_QWORD **)(a1 - 16) = result;
    if ( result )
      result[2] = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a2;
  if ( a2 )
  {
    result = *(_QWORD **)(a2 + 16);
    *(_QWORD *)(a1 - 24) = result;
    if ( result )
      result[2] = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
  return result;
}
