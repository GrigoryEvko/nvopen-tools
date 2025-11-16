// Function: sub_27CA920
// Address: 0x27ca920
//
_QWORD *__fastcall sub_27CA920(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r15
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  _QWORD *v9; // r13
  __int64 *v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v18[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = (__int64)a1[1];
  v3 = *(_QWORD *)(**a1 + 48);
  v4 = sub_D95540(a2);
  v5 = sub_DA2C50(v2, v4, 0, 0);
  v6 = (__int64)a1[1];
  v7 = (__int64)v5;
  v8 = sub_D95540(a2);
  v9 = sub_DA2C50(v6, v8, 1, 0);
  if ( !(unsigned __int8)sub_F705A0(a2, v3, a1[1]) )
  {
    if ( (unsigned __int8)sub_F70530(a2, v3, a1[1]) )
    {
      return (_QWORD *)v7;
    }
    else
    {
      v11 = sub_DCAF50(a1[1], (__int64)v9, 0);
      v12 = a1[1];
      v16 = (__int64)v11;
      v14 = sub_DCE160(v12, a2, v7, v13);
      v18[0] = sub_DCDFA0(v12, v14, v16, v15);
      v18[1] = v9;
      v17[0] = (unsigned __int64)v18;
      v17[1] = 0x200000002LL;
      v9 = sub_DC7EB0(v12, (__int64)v17, 0, 0);
      if ( (_QWORD *)v17[0] != v18 )
        _libc_free(v17[0]);
    }
  }
  return v9;
}
