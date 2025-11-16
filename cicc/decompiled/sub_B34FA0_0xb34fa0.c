// Function: sub_B34FA0
// Address: 0xb34fa0
//
__int64 __fastcall sub_B34FA0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // r12
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 *v15; // rax
  int v16; // [rsp+4h] [rbp-5Ch] BYREF
  __int64 **v17; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v18[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( !a6 )
    a6 = sub_ACADE0(a2);
  v18[2] = a6;
  v18[1] = a5;
  v17 = a2;
  v18[0] = a3;
  v10 = sub_B34BE0(a1, 0xE2u, (int)v18, 3, (__int64)&v17, 1, a7);
  if ( BYTE1(a4) )
  {
    v12 = (__int64 *)sub_BD5C60(v10, 226);
    v13 = a4;
    v16 = 0;
    v14 = sub_A77A40(v12, a4);
    v15 = (__int64 *)sub_BD5C60(v10, v13);
    *(_QWORD *)(v10 + 72) = sub_A7B660((__int64 *)(v10 + 72), v15, &v16, 1, v14);
  }
  return v10;
}
