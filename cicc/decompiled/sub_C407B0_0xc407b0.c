// Function: sub_C407B0
// Address: 0xc407b0
//
__int64 __fastcall sub_C407B0(_QWORD *a1, __int64 *a2, _DWORD *a3)
{
  _DWORD *v4; // rax
  _DWORD *v6; // rbx
  __int64 v7; // [rsp+8h] [rbp-98h]
  void *v8[4]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE v9[32]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v10[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = sub_C33340();
  if ( a3 != v4 )
    return sub_C338E0((__int64)a1, (__int64)a2);
  v6 = v4;
  v7 = *a2;
  sub_C338E0((__int64)v9, (__int64)a2);
  sub_C338E0((__int64)v10, (__int64)v9);
  sub_C407B0(v8, v10, v7);
  sub_C338F0((__int64)v10);
  if ( v6 == dword_3F657A0 )
    sub_C3C460(v10, (__int64)v6);
  else
    sub_C37380(v10, (__int64)dword_3F657A0);
  sub_C3C930(a1, (__int64)v6, v8, v10);
  if ( v6 == (_DWORD *)v10[0] )
    sub_969EE0((__int64)v10);
  else
    sub_C338F0((__int64)v10);
  if ( v6 == v8[0] )
    sub_969EE0((__int64)v8);
  else
    sub_C338F0((__int64)v8);
  return sub_C338F0((__int64)v9);
}
