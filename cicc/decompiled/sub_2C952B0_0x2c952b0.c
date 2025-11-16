// Function: sub_2C952B0
// Address: 0x2c952b0
//
_QWORD *__fastcall sub_2C952B0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // r9
  _BYTE *v6; // rdi
  __int64 v7; // rax
  _QWORD *result; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD *v11; // [rsp+8h] [rbp-98h]
  _BYTE *v12; // [rsp+20h] [rbp-80h] BYREF
  __int64 v13; // [rsp+28h] [rbp-78h]
  _BYTE v14[112]; // [rsp+30h] [rbp-70h] BYREF

  v11 = (_QWORD *)a1;
  if ( sub_D968A0(a1) )
  {
    v7 = sub_D95540(a1);
    v10 = sub_DA2C50((__int64)a3, v7, 0, 0);
    goto LABEL_8;
  }
  v12 = v14;
  v13 = 0x800000000LL;
  v4 = sub_D95540(a1);
  v10 = sub_DA2C50((__int64)a3, v4, 0, 0);
  sub_2C94930(a1, 0, (__int64)&v12, a3, (__int64)&v10, v5);
  if ( !(_DWORD)v13 )
  {
    v11 = 0;
    v6 = v12;
    if ( v12 == v14 )
      goto LABEL_8;
    goto LABEL_12;
  }
  if ( (unsigned int)v13 == 1 )
  {
    v6 = v12;
    v11 = *(_QWORD **)v12;
  }
  else
  {
    v9 = sub_DC7EB0(a3, (__int64)&v12, 0, 0);
    v6 = v12;
    v11 = v9;
  }
  if ( v6 != v14 )
LABEL_12:
    _libc_free((unsigned __int64)v6);
LABEL_8:
  sub_2C95190(&v12, a2, a3);
  result = 0;
  if ( v11 == (_QWORD *)v13 )
    return sub_DCC810(a3, (__int64)v10, (__int64)v12, 0, 0);
  return result;
}
