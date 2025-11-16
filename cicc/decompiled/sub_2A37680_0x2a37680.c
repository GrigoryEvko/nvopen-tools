// Function: sub_2A37680
// Address: 0x2a37680
//
_BYTE *__fastcall sub_2A37680(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // rdx
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r12
  _BYTE *result; // rax
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-58h]
  __int8 *v12[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF

  v2 = ((__int64 (__fastcall *)(__int64 **, __int64))(*a1)[3])(a1, 1);
  v3 = (__int64)a1[2];
  v4 = v2;
  v11 = v5;
  v6 = ((__int64 (__fastcall *)(__int64 **))(*a1)[4])(a1);
  if ( v6 == 14 )
  {
    v7 = sub_22077B0(0x1B0u);
    v8 = v7;
    if ( v7 )
      sub_B176B0(v7, v3, v4, v11, a2);
  }
  else
  {
    if ( v6 != 15 )
      BUG();
    v10 = sub_22077B0(0x1B0u);
    v8 = v10;
    if ( v10 )
      sub_B178C0(v10, v3, v4, v11, a2);
  }
  ((void (__fastcall *)(__int8 **, __int64 **, const char *, __int64))(*a1)[2])(v12, a1, "Initialization", 14);
  sub_B18290(v8, v12[0], (size_t)v12[1]);
  if ( (__int64 *)v12[0] != &v13 )
    j_j___libc_free_0((unsigned __int64)v12[0]);
  result = sub_1049740(a1[1], v8);
  if ( v8 )
    return (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8);
  return result;
}
