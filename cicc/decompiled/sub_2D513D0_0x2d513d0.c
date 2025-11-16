// Function: sub_2D513D0
// Address: 0x2d513d0
//
_QWORD *__fastcall sub_2D513D0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi

  v3 = (__int64)(a1 + 1);
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 888LL);
  *(_QWORD *)(v3 - 8) = v4;
  sub_C7C840(v3, v4, 1, 35);
  a1[9] = 0;
  a1[11] = 0xA000000000LL;
  a1[14] = 0x9800000000LL;
  a1[17] = 0x1800000000LL;
  a1[10] = 0;
  a1[12] = 0;
  a1[13] = 0;
  a1[15] = 0;
  a1[16] = 0;
  return a1;
}
