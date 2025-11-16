// Function: sub_38AF940
// Address: 0x38af940
//
__int64 __fastcall sub_38AF940(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  int v7; // eax
  unsigned __int64 v8; // rsi
  unsigned int v9; // r12d
  __int64 v10; // rax
  unsigned int v12; // r15d
  _QWORD *v13; // r13
  __int64 *v14; // [rsp+8h] [rbp-C8h]
  __int64 v15; // [rsp+10h] [rbp-C0h]
  __int64 *v16; // [rsp+18h] [rbp-B8h]
  __int64 *v17; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE v18[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v19; // [rsp+40h] [rbp-90h]
  const char *v20; // [rsp+50h] [rbp-80h] BYREF
  __int64 v21; // [rsp+58h] [rbp-78h]
  _BYTE v22[112]; // [rsp+60h] [rbp-70h] BYREF

  v17 = 0;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 55, "expected 'within' after catchpad") )
    return 1;
  v7 = *((_DWORD *)a1 + 16);
  if ( v7 != 369 && v7 != 375 )
  {
    v8 = (unsigned __int64)a1[7];
    v22[1] = 1;
    v20 = "expected scope value for catchpad";
    v22[0] = 3;
    return (unsigned int)sub_38814C0((__int64)(a1 + 1), v8, (__int64)&v20);
  }
  v10 = sub_16432D0(*a1);
  if ( (unsigned __int8)sub_38A1070(a1, v10, &v17, a3, a4, a5, a6) )
    return 1;
  v20 = v22;
  v21 = 0x800000000LL;
  v9 = sub_38AF7F0((__int64)a1, (__int64)&v20, a3, a4, a5, a6);
  if ( !(_BYTE)v9 )
  {
    v19 = 257;
    v15 = (unsigned int)v21;
    v12 = v21 + 1;
    v14 = (__int64 *)v20;
    v16 = v17;
    v13 = sub_1648A60(56, (int)v21 + 1);
    if ( v13 )
      sub_15F8230((__int64)v13, 50, v16, v14, v15, v12, (__int64)v18, 0);
    *a2 = v13;
  }
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v9;
}
