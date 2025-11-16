// Function: sub_2C95190
// Address: 0x2c95190
//
void __fastcall sub_2C95190(_QWORD *a1, __int64 a2, __int64 *a3)
{
  bool v5; // al
  __int64 v6; // rax
  __int64 v7; // r9
  _BYTE *v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // [rsp+0h] [rbp-80h] BYREF
  __int64 v12; // [rsp+8h] [rbp-78h]
  _BYTE v13[112]; // [rsp+10h] [rbp-70h] BYREF

  v5 = sub_D968A0(a2);
  a1[1] = a2;
  if ( v5 )
  {
    v9 = sub_D95540(a2);
    *a1 = sub_DA2C50((__int64)a3, v9, 0, 0);
    return;
  }
  v11 = v13;
  v12 = 0x800000000LL;
  v6 = sub_D95540(a2);
  *a1 = sub_DA2C50((__int64)a3, v6, 0, 0);
  sub_2C94930(a2, 0, (__int64)&v11, a3, (__int64)a1, v7);
  if ( !(_DWORD)v12 )
  {
    v8 = v11;
    a1[1] = 0;
    if ( v8 == v13 )
      return;
LABEL_10:
    _libc_free((unsigned __int64)v8);
    return;
  }
  if ( (unsigned int)v12 == 1 )
  {
    v8 = v11;
    a1[1] = *v11;
  }
  else
  {
    v10 = sub_DC7EB0(a3, (__int64)&v11, 0, 0);
    v8 = v11;
    a1[1] = v10;
  }
  if ( v8 != v13 )
    goto LABEL_10;
}
