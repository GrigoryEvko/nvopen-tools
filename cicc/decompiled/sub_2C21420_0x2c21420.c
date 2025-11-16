// Function: sub_2C21420
// Address: 0x2c21420
//
__int64 *__fastcall sub_2C21420(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v9[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v9[0] = *(_QWORD *)(a1 + 88);
  if ( v9[0] )
    sub_2AAAFA0(v9);
  sub_2BF1A90(a2, (__int64)v9);
  sub_9C6650(v9);
  v4 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v5 = *(__int64 **)(a2 + 904);
  v6 = *(_QWORD *)(v4 + 8);
  v10 = 260;
  v9[0] = a1 + 152;
  v7 = sub_D5C860(v5, v6, 2, (__int64)v9);
  return sub_2BF26E0(a2, a1 + 96, v7, 0);
}
