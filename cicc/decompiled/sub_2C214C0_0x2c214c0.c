// Function: sub_2C214C0
// Address: 0x2c214c0
//
__int64 *__fastcall sub_2C214C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 *v6; // rdi
  __int64 v7; // r13
  unsigned __int8 *v8; // rsi
  __int64 v10; // [rsp+8h] [rbp-68h]
  __int64 v11[4]; // [rsp+10h] [rbp-60h] BYREF
  char v12; // [rsp+30h] [rbp-40h]
  char v13; // [rsp+31h] [rbp-3Fh]

  v10 = sub_2BF3650(a2 + 96, a1);
  v4 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v13 = 1;
  v5 = v4;
  v12 = 3;
  v6 = *(__int64 **)(a2 + 904);
  v11[0] = (__int64)"active.lane.mask";
  v7 = sub_D5C860(v6, *(_QWORD *)(v4 + 8), 2, (__int64)v11);
  sub_F0A850(v7, v5, v10);
  v11[0] = *(_QWORD *)(a1 + 88);
  if ( v11[0] )
    sub_2AAAFA0(v11);
  if ( (__int64 *)(v7 + 48) != v11 )
  {
    sub_9C6650((_QWORD *)(v7 + 48));
    v8 = (unsigned __int8 *)v11[0];
    *(_QWORD *)(v7 + 48) = v11[0];
    if ( v8 )
    {
      sub_B976B0((__int64)v11, v8, v7 + 48);
      v11[0] = 0;
    }
  }
  sub_9C6650(v11);
  return sub_2BF26E0(a2, a1 + 96, v7, 0);
}
