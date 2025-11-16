// Function: sub_257AC00
// Address: 0x257ac00
//
__int64 __fastcall sub_257AC00(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  char v11; // [rsp+1Fh] [rbp-91h] BYREF
  __int64 v12; // [rsp+20h] [rbp-90h] BYREF
  __int64 v13; // [rsp+28h] [rbp-88h] BYREF
  __int64 v14; // [rsp+34h] [rbp-7Ch] BYREF
  int v15; // [rsp+3Ch] [rbp-74h]
  _QWORD v16[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v17; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v18; // [rsp+68h] [rbp-48h]
  __int64 v19; // [rsp+70h] [rbp-40h]
  __int64 *v20; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(sub_B43CB0(*a3) + 80);
  if ( !v7 )
    BUG();
  v8 = *(_QWORD *)(v7 + 32);
  if ( v8 )
    v8 -= 24;
  v12 = v8;
  if ( *a3 != v8 )
  {
    if ( !(unsigned __int8)sub_257ADC0(a1, a2, v8, a3[1], 0) )
      return sub_2576E80(a1, a2, 0, (__int64)a3, 0, a4);
    v8 = *a3;
  }
  v16[0] = a2;
  v16[1] = a1;
  v16[2] = a3;
  v16[3] = &v12;
  v9 = sub_B43CB0(v8);
  sub_250D230((unsigned __int64 *)&v17, v9, 4, 0);
  v13 = sub_25285C0(a2, (__int64)v17, (__int64)v18, a1, 1, 0, 1);
  v17 = v16;
  v18 = &v13;
  v14 = 0xB00000005LL;
  v19 = a2;
  v20 = a3;
  v11 = 0;
  v15 = 56;
  if ( (unsigned __int8)sub_2526370(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25A0CD0,
                          (__int64)&v17,
                          a1,
                          (int *)&v14,
                          3,
                          &v11,
                          1,
                          0) )
    return sub_2576E80(a1, a2, 0, (__int64)a3, 1, a4);
  else
    return sub_2576E80(a1, a2, 1, (__int64)a3, 1, a4);
}
