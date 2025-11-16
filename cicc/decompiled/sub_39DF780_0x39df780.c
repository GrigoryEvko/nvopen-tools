// Function: sub_39DF780
// Address: 0x39df780
//
void __fastcall sub_39DF780(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 v5[2]; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE v6[16]; // [rsp+20h] [rbp-D0h] BYREF
  void *v7; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v8; // [rsp+38h] [rbp-B8h]
  __int64 v9; // [rsp+40h] [rbp-B0h]
  __int64 v10; // [rsp+48h] [rbp-A8h]
  int v11; // [rsp+50h] [rbp-A0h]
  __int64 *v12; // [rsp+58h] [rbp-98h]
  _QWORD v13[18]; // [rsp+60h] [rbp-90h] BYREF

  sub_16E4AB0((__int64)v13, a1, 0, 70);
  nullsub_622();
  if ( (unsigned __int8)sub_16E4B20() )
  {
    if ( !(unsigned __int8)sub_16E3400(v13, 0) )
    {
      v7 = 0;
      v8 = 0;
      sub_16E4FB0((__int64)v13, (__int64 *)&v7);
      sub_16E4080((__int64)v13);
      BUG();
    }
    v5[0] = (unsigned __int64)v6;
    v5[1] = 0;
    v7 = &unk_49EFBE0;
    v6[0] = 0;
    v11 = 1;
    v10 = 0;
    v9 = 0;
    v8 = 0;
    v12 = (__int64 *)v5;
    sub_16E4080((__int64)v13);
    sub_155BB10(a2, (__int64)&v7, 0, 0, 0, a3);
    if ( v10 != v8 )
      sub_16E7BA0((__int64 *)&v7);
    v3 = v12[1];
    v4[0] = *v12;
    v4[1] = v3;
    sub_16E4FB0((__int64)v13, v4);
    sub_16E7BC0((__int64 *)&v7);
    if ( (_BYTE *)v5[0] != v6 )
      j_j___libc_free_0(v5[0]);
    nullsub_628();
  }
  sub_16E4BA0((__int64)v13);
  sub_16E3E40(v13);
}
