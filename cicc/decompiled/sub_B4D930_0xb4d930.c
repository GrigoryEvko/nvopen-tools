// Function: sub_B4D930
// Address: 0xb4d930
//
_QWORD *__fastcall sub_B4D930(__int64 a1, __int64 a2, __int16 a3, char a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v10; // rax
  _QWORD *result; // rax
  __int16 v12; // dx

  v10 = sub_BCB120(a2);
  result = sub_B44260(a1, v10, 35, 0, a5, a6);
  v12 = *(_WORD *)(a1 + 2);
  *(_BYTE *)(a1 + 72) = a4;
  *(_WORD *)(a1 + 2) = a3 | v12 & 0xFFF8;
  return result;
}
