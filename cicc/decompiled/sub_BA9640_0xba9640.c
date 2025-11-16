// Function: sub_BA9640
// Address: 0xba9640
//
_QWORD *__fastcall sub_BA9640(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rsi

  result = a1;
  v3 = a2 + 8;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = a2 + 24;
  v6 = *(_QWORD *)(a2 + 32);
  result[2] = v3;
  *result = v4;
  result[1] = v6;
  result[3] = v5;
  result[4] = v3;
  result[5] = v5;
  result[6] = v3;
  result[7] = v5;
  return result;
}
