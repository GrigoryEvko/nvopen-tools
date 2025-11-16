// Function: sub_E81A90
// Address: 0xe81a90
//
unsigned __int64 __fastcall sub_E81A90(__int64 a1, _QWORD *a2, char a3, int a4)
{
  __int64 v6; // rdx
  unsigned __int64 result; // rax
  int v8; // edx

  v6 = a2[24];
  a2[34] += 24LL;
  result = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2[25] >= result + 24 && v6 )
    a2[24] = result + 24;
  else
    result = sub_9D1E70((__int64)(a2 + 24), 24, 24, 3);
  v8 = a4;
  *(_BYTE *)result = 1;
  *(_QWORD *)(result + 8) = 0;
  if ( a3 )
  {
    BYTE1(v8) = BYTE1(a4) | 1;
    a4 = v8;
  }
  *(_QWORD *)(result + 16) = a1;
  *(_WORD *)(result + 1) = a4;
  *(_BYTE *)(result + 3) = BYTE2(a4);
  return result;
}
