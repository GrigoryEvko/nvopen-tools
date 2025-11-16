// Function: sub_E82E60
// Address: 0xe82e60
//
__int64 __fastcall sub_E82E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 v6; // r12
  __int64 result; // rax

  v6 = a3;
  result = sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, a5, a6);
  *(_WORD *)(a2 + 12) = v6;
  return result;
}
