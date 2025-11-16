// Function: sub_2208C50
// Address: 0x2208c50
//
_QWORD *__fastcall sub_2208C50(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rcx
  _QWORD *v4; // rax
  _QWORD *result; // rax

  if ( a1 != a3 )
  {
    v3 = *(_QWORD **)(a3 + 8);
    v4 = *(_QWORD **)(a2 + 8);
    *v3 = a1;
    *v4 = a3;
    result = *(_QWORD **)(a1 + 8);
    *result = a2;
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a3 + 8) = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a2 + 8) = result;
  }
  return result;
}
