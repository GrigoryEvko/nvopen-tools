// Function: sub_131D410
// Address: 0x131d410
//
__int64 __fastcall sub_131D410(__int64 a1, _WORD *a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax

  v6 = *a4 + (unsigned __int16)(8 * *a2);
  v7 = a3 + *a4;
  *a4 = v6;
  result = a3 + v6;
  *(_QWORD *)a1 = result;
  *(_WORD *)(a1 + 16) = result;
  *(_WORD *)(a1 + 18) = v7;
  *(_WORD *)(a1 + 20) = result;
  return result;
}
