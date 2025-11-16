// Function: sub_2673A60
// Address: 0x2673a60
//
__int64 __fastcall sub_2673A60(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v11; // rax
  __int64 v12; // r15
  int v14; // [rsp+Ch] [rbp-44h]

  v14 = a4 + 1;
  v11 = sub_BD2C40(88, (int)a4 + 1);
  v12 = (__int64)v11;
  if ( v11 )
  {
    sub_B44260((__int64)v11, **(_QWORD **)(a1 + 16), 56, v14 & 0x7FFFFFF, a7, a8);
    *(_QWORD *)(v12 + 72) = 0;
    sub_B4A290(v12, a1, a2, a3, a4, a5, 0, 0);
  }
  return v12;
}
