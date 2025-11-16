// Function: sub_63BA50
// Address: 0x63ba50
//
_QWORD *__fastcall sub_63BA50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rax
  _QWORD *result; // rax
  __int64 v13; // rdx

  v8 = (_QWORD *)sub_724D50(9);
  v9 = *(_QWORD *)dword_4F07508;
  v8[22] = a1;
  v8[16] = a3;
  v8[8] = v9;
  v10 = (_QWORD *)sub_724D50(10);
  v11 = *(_QWORD *)dword_4F07508;
  v10[16] = a2;
  v10[8] = v11;
  *(_QWORD *)(a4 + 56) = v10;
  result = (_QWORD *)sub_724D50(11);
  v13 = v8[8];
  result[23] = a5;
  result[8] = v13;
  result[22] = v8;
  v10[22] = result;
  v10[23] = result;
  return result;
}
