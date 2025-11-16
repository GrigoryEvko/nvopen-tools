// Function: sub_1002C90
// Address: 0x1002c90
//
__int64 __fastcall sub_1002C90(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rsi

  v4 = a2[1];
  result = a1;
  v6 = a2[2];
  v7 = a2[5];
  v8 = a2[6];
  *(_QWORD *)result = a3;
  *(_QWORD *)(result + 32) = v4;
  *(_QWORD *)(result + 8) = v7;
  *(_QWORD *)(result + 16) = v8;
  *(_QWORD *)(result + 24) = v6;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_WORD *)(result + 64) = 257;
  return result;
}
