// Function: sub_371CDC0
// Address: 0x371cdc0
//
__int64 __fastcall sub_371CDC0(
        unsigned int a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v8; // r13
  __int64 v10; // r12
  _QWORD *v11; // rax
  __int64 v12; // r15
  __int64 v14; // [rsp+8h] [rbp-58h]
  unsigned __int16 v16; // [rsp+1Eh] [rbp-42h]
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int16 v18; // [rsp+28h] [rbp-38h]

  v8 = 0;
  v10 = sub_B6E160(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a7 + 40) + 72LL) + 40LL), a1, a2, a3);
  sub_B43C00((__int64)&v17, a7);
  if ( v10 )
    v8 = *(_QWORD *)(v10 + 24);
  v14 = v17;
  v16 = v18;
  v11 = sub_BD2C40(88, (int)a5 + 1);
  v12 = (__int64)v11;
  if ( v11 )
  {
    sub_B44260((__int64)v11, **(_QWORD **)(v8 + 16), 56, (a5 + 1) & 0x7FFFFFF, v14, v16);
    *(_QWORD *)(v12 + 72) = 0;
    sub_B4A290(v12, v8, v10, a4, a5, a6, 0, 0);
  }
  return v12;
}
