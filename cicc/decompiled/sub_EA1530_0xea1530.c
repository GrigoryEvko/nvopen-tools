// Function: sub_EA1530
// Address: 0xea1530
//
__int64 __fastcall sub_EA1530(unsigned int a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi

  v3 = 0;
  if ( a2 )
  {
    a1 += 8;
    v3 = 8;
  }
  v4 = a3[24];
  v5 = a1;
  a3[34] += a1;
  v6 = (v4 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v7 = a1 + v6;
  if ( a3[25] < v7 || !v4 )
    return v3 + sub_9D1E70((__int64)(a3 + 24), v5, v5, 3);
  a3[24] = v7;
  return v3 + v6;
}
