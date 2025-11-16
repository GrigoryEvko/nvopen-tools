// Function: sub_10DF260
// Address: 0x10df260
//
__int64 __fastcall sub_10DF260(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 **v5; // rdi
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // r12
  int v10; // [rsp+Ch] [rbp-64h] BYREF
  __int64 v11[2]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE v12[32]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v13; // [rsp+40h] [rbp-30h]

  v4 = sub_ACADE0(*(__int64 ***)(a2 + 8));
  v5 = *(__int64 ***)(a1 + 8);
  v11[1] = a3;
  v11[0] = v4;
  v10 = 0;
  v6 = sub_AD24A0(v5, v11, 2);
  v13 = 257;
  v7 = sub_BD2C40(104, unk_3F148BC);
  v8 = (__int64)v7;
  if ( v7 )
  {
    sub_B44260((__int64)v7, *(_QWORD *)(v6 + 8), 65, 2u, 0, 0);
    *(_QWORD *)(v8 + 72) = v8 + 88;
    *(_QWORD *)(v8 + 80) = 0x400000000LL;
    sub_B4FD20(v8, v6, a2, &v10, 1, (__int64)v12);
  }
  return v8;
}
