// Function: sub_2C215B0
// Address: 0x2c215b0
//
__int64 *__fastcall sub_2C215B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned __int8 *v11; // rsi
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v4 = sub_2BF3650(a2 + 96, a1);
  v5 = *(_DWORD *)(a1 + 56);
  BYTE4(v14[0]) = 0;
  v6 = 0;
  LODWORD(v14[0]) = 0;
  if ( v5 )
    v6 = **(_QWORD **)(a1 + 48);
  v13 = v4;
  v7 = sub_2BFB120(a2, v6, (unsigned int *)v14);
  v8 = *(__int64 **)(a2 + 904);
  v9 = v7;
  v15 = 260;
  v14[0] = a1 + 152;
  v10 = sub_D5C860(v8, *(_QWORD *)(v7 + 8), 2, (__int64)v14);
  sub_F0A850(v10, v9, v13);
  v14[0] = *(_QWORD *)(a1 + 88);
  if ( v14[0] )
    sub_2AAAFA0(v14);
  if ( (__int64 *)(v10 + 48) != v14 )
  {
    sub_9C6650((_QWORD *)(v10 + 48));
    v11 = (unsigned __int8 *)v14[0];
    *(_QWORD *)(v10 + 48) = v14[0];
    if ( v11 )
    {
      sub_B976B0((__int64)v14, v11, v10 + 48);
      v14[0] = 0;
    }
  }
  sub_9C6650(v14);
  return sub_2BF26E0(a2, a1 + 96, v10, 1);
}
