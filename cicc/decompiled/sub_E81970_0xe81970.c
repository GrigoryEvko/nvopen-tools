// Function: sub_E81970
// Address: 0xe81970
//
unsigned __int64 __fastcall sub_E81970(int a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // rcx
  unsigned __int64 result; // rax

  v5 = a3[24];
  a3[34] += 24LL;
  result = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3[25] >= result + 24 && v5 )
    a3[24] = result + 24;
  else
    result = sub_9D1E70((__int64)(a3 + 24), 24, 24, 3);
  *(_WORD *)(result + 1) = a1;
  *(_BYTE *)result = 3;
  *(_BYTE *)(result + 3) = BYTE2(a1);
  *(_QWORD *)(result + 8) = a4;
  *(_QWORD *)(result + 16) = a2;
  return result;
}
