// Function: sub_12F9390
// Address: 0x12f9390
//
__int64 __fastcall sub_12F9390(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // r12d
  __int64 v5; // r13
  struct __jmp_buf_tag *v6; // r12
  int v7; // eax
  void *v8; // rdi
  _QWORD v10[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v12[4]; // [rsp+30h] [rbp-50h] BYREF
  int v13; // [rsp+50h] [rbp-30h]
  _QWORD *v14; // [rsp+58h] [rbp-28h]

  sub_1C3E900();
  v2 = sub_163A1D0(a1, a2);
  sub_15A5150(v2);
  sub_1CCF750(v2);
  v10[0] = v11;
  v10[1] = 0;
  LOBYTE(v11[0]) = 0;
  v13 = 1;
  memset(&v12[1], 0, 24);
  v12[0] = &unk_49EFBE0;
  v14 = v10;
  if ( (unsigned __int8)sub_166CBC0(a1, v12, 0) )
  {
    v3 = 6;
    sub_1C3EFD0(v10, 1);
    sub_1C3E9C0(a2);
    goto LABEL_3;
  }
  v5 = sub_1C3E710();
  v6 = (struct __jmp_buf_tag *)sub_16D40F0(v5);
  if ( !v6 )
  {
    v8 = (void *)sub_1C42D70(200, 8);
    memset(v8, 0, 0xC8u);
    sub_16D40E0(v5, v8);
    v6 = (struct __jmp_buf_tag *)sub_16D40F0(v5);
  }
  v7 = _setjmp(v6);
  if ( v7 )
  {
    if ( v7 == 1 )
    {
      v3 = 6;
      sub_1C3E9C0(a2);
      goto LABEL_3;
    }
  }
  else
  {
    sub_1C3DD40(a1, 0, 0, 0, 0, 0);
  }
  v3 = 0;
  sub_1C3E9C0(a2);
LABEL_3:
  sub_16E7BC0(v12);
  if ( (_QWORD *)v10[0] != v11 )
    j_j___libc_free_0(v10[0], v11[0] + 1LL);
  return v3;
}
