// Function: sub_E81A00
// Address: 0xe81a00
//
unsigned __int64 __fastcall sub_E81A00(int a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // rdx
  unsigned __int64 result; // rax

  v7 = a4[24];
  a4[34] += 32LL;
  result = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a4[25] >= result + 32 && v7 )
    a4[24] = result + 32;
  else
    result = sub_9D1E70((__int64)(a4 + 24), 32, 32, 3);
  *(_WORD *)(result + 1) = a1;
  *(_BYTE *)result = 0;
  *(_BYTE *)(result + 3) = BYTE2(a1);
  *(_QWORD *)(result + 8) = a5;
  *(_QWORD *)(result + 16) = a2;
  *(_QWORD *)(result + 24) = a3;
  return result;
}
