// Function: sub_BA96F0
// Address: 0xba96f0
//
_QWORD *__fastcall sub_BA96F0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r11
  __int64 v3; // r10
  _QWORD *result; // rax
  _QWORD *v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rdi
  _QWORD *v8; // rcx
  _QWORD *v9; // rdx
  __int64 v10; // rsi

  v2 = a2[8];
  v3 = a2[6];
  result = a1;
  v5 = a2 + 7;
  v6 = a2[2];
  v7 = a2 + 5;
  v8 = a2 + 1;
  v9 = a2 + 3;
  v10 = a2[4];
  *result = v2;
  result[1] = v3;
  result[2] = v6;
  result[3] = v10;
  result[4] = v5;
  result[5] = v7;
  result[6] = v8;
  result[7] = v9;
  result[8] = v5;
  result[9] = v7;
  result[10] = v8;
  result[11] = v9;
  result[12] = v5;
  result[13] = v7;
  result[14] = v8;
  result[15] = v9;
  return result;
}
