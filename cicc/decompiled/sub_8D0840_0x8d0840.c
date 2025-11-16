// Function: sub_8D0840
// Address: 0x8d0840
//
__int64 __fastcall sub_8D0840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 result; // rax
  __int64 v9; // rsi
  bool v10; // zf

  v7 = a2;
  result = sub_822B10(40, a2, a3, a4, a5, a6);
  v9 = qword_4F60530;
  v10 = qword_4F60540 == 0;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = a1;
  *(_QWORD *)(result + 16) = v7;
  *(_QWORD *)(result + 24) = v9;
  *(_QWORD *)(result + 32) = a3;
  if ( v10 )
    qword_4F60540 = result;
  if ( qword_4F60538 )
    *(_QWORD *)qword_4F60538 = result;
  qword_4F60538 = result;
  if ( (v7 & 7) != 0 )
  {
    result = (int)(8 - (v7 & 7));
    v7 += result;
  }
  qword_4F60530 = v7 + v9;
  return result;
}
