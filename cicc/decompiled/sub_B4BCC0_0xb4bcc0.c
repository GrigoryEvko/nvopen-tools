// Function: sub_B4BCC0
// Address: 0xb4bcc0
//
_QWORD *__fastcall sub_B4BCC0(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *result; // rax

  v6 = sub_BD5C60(a2, a2);
  v7 = sub_BCB120(v6);
  result = sub_B44260(a1, v7, 6, 1u, a3, a4);
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
