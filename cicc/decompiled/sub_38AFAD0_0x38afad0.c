// Function: sub_38AFAD0
// Address: 0x38afad0
//
__int64 __fastcall sub_38AFAD0(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  int v7; // eax
  __int64 v8; // rax
  unsigned int v9; // r12d
  const char *v10; // rdi
  unsigned int v12; // r15d
  _QWORD *v13; // r13
  unsigned __int64 v14; // rsi
  __int64 *v15; // [rsp+8h] [rbp-C8h]
  __int64 v16; // [rsp+10h] [rbp-C0h]
  __int64 *v17; // [rsp+18h] [rbp-B8h]
  __int64 *v18; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v19[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v20; // [rsp+40h] [rbp-90h]
  const char *v21; // [rsp+50h] [rbp-80h] BYREF
  __int64 v22; // [rsp+58h] [rbp-78h]
  _BYTE v23[112]; // [rsp+60h] [rbp-70h] BYREF

  v18 = 0;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 55, "expected 'within' after cleanuppad") )
    return 1;
  v7 = *((_DWORD *)a1 + 16);
  if ( v7 != 52 && v7 != 375 && v7 != 369 )
  {
    v14 = (unsigned __int64)a1[7];
    v23[1] = 1;
    v21 = "expected scope value for cleanuppad";
    v23[0] = 3;
    return (unsigned int)sub_38814C0((__int64)(a1 + 1), v14, (__int64)&v21);
  }
  v8 = sub_16432D0(*a1);
  if ( (unsigned __int8)sub_38A1070(a1, v8, &v18, a3, a4, a5, a6) )
    return 1;
  v21 = v23;
  v22 = 0x800000000LL;
  v9 = sub_38AF7F0((__int64)a1, (__int64)&v21, a3, a4, a5, a6);
  if ( (_BYTE)v9 )
  {
    v10 = v21;
    if ( v21 != v23 )
LABEL_7:
      _libc_free((unsigned __int64)v10);
  }
  else
  {
    v20 = 257;
    v16 = (unsigned int)v22;
    v12 = v22 + 1;
    v15 = (__int64 *)v21;
    v17 = v18;
    v13 = sub_1648A60(56, (int)v22 + 1);
    if ( v13 )
      sub_15F8230((__int64)v13, 49, v17, v15, v16, v12, (__int64)v19, 0);
    v10 = v21;
    *a2 = v13;
    if ( v10 != v23 )
      goto LABEL_7;
  }
  return v9;
}
