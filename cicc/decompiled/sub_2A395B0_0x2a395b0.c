// Function: sub_2A395B0
// Address: 0x2a395b0
//
void __fastcall sub_2A395B0(__int64 a1, unsigned __int8 *a2, char a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int8 *v8[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v10[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v11[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v12; // [rsp+50h] [rbp-60h]
  _QWORD v13[10]; // [rsp+60h] [rbp-50h] BYREF

  sub_B18290(a4, "Call to ", 8u);
  if ( !a3 )
  {
    sub_B16430((__int64)v10, "UnknownLibCall", 0xEu, "unknown", 7);
    v7 = sub_2A38130(a4, (__int64)v10);
    sub_B18290(v7, " function ", 0xAu);
    if ( v12 != v13 )
      j_j___libc_free_0((unsigned __int64)v12);
    if ( (_QWORD *)v10[0] != v11 )
      j_j___libc_free_0(v10[0]);
  }
  sub_B16080((__int64)v10, "Callee", 6, a2);
  v6 = sub_2A38130(a4, (__int64)v10);
  (*(void (__fastcall **)(__int8 **, __int64, const char *, _QWORD))(*(_QWORD *)a1 + 16LL))(v8, a1, byte_3F871B3, 0);
  sub_B18290(v6, v8[0], (size_t)v8[1]);
  if ( (__int64 *)v8[0] != &v9 )
    j_j___libc_free_0((unsigned __int64)v8[0]);
  if ( v12 != v13 )
    j_j___libc_free_0((unsigned __int64)v12);
  if ( (_QWORD *)v10[0] != v11 )
    j_j___libc_free_0(v10[0]);
}
