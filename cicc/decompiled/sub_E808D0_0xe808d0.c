// Function: sub_E808D0
// Address: 0xe808d0
//
unsigned __int64 __fastcall sub_E808D0(__int64 a1, unsigned __int16 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 v7; // r12
  __int64 v9; // rax

  v4 = a4;
  v5 = a3[24];
  a3[34] += 24LL;
  v6 = a3[19];
  v7 = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3[25] >= v7 + 24 && v5 )
  {
    a3[24] = v7 + 24;
  }
  else
  {
    v9 = sub_9D1E70((__int64)(a3 + 24), 24, 24, 3);
    v4 = a4;
    v7 = v9;
  }
  sub_E807A0(v7, a1, a2, v6, v4);
  return v7;
}
