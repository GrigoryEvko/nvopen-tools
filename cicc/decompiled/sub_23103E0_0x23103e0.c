// Function: sub_23103E0
// Address: 0x23103e0
//
_BYTE *__fastcall sub_23103E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  _BYTE *result; // rax

  v6 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) <= 5 )
  {
    sub_CB6200(a2, "cgscc(", 6u);
  }
  else
  {
    *(_DWORD *)v6 = 1668507491;
    *(_WORD *)(v6 + 4) = 10339;
    *(_QWORD *)(a2 + 32) += 6LL;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
    *(_QWORD *)(a1 + 8),
    a2,
    a3,
    a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
