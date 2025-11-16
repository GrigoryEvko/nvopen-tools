// Function: sub_2A39010
// Address: 0x2a39010
//
void __fastcall sub_2A39010(__int64 a1, _BYTE *a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int8 *v10[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v11; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v13[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v14; // [rsp+50h] [rbp-60h]
  _QWORD v15[10]; // [rsp+60h] [rbp-50h] BYREF

  sub_B18290(a5, "Call to ", 8u);
  if ( !a4 )
  {
    sub_B16430((__int64)v12, "UnknownLibCall", 0xEu, "unknown", 7);
    v9 = sub_2A38130(a5, (__int64)v12);
    sub_B18290(v9, " function ", 0xAu);
    if ( v14 != v15 )
      j_j___libc_free_0((unsigned __int64)v14);
    if ( (_QWORD *)v12[0] != v13 )
      j_j___libc_free_0(v12[0]);
  }
  sub_B16430((__int64)v12, "Callee", 6u, a2, a3);
  v8 = sub_2A38130(a5, (__int64)v12);
  (*(void (__fastcall **)(__int8 **, __int64, const char *, _QWORD))(*(_QWORD *)a1 + 16LL))(v10, a1, byte_3F871B3, 0);
  sub_B18290(v8, v10[0], (size_t)v10[1]);
  if ( (__int64 *)v10[0] != &v11 )
    j_j___libc_free_0((unsigned __int64)v10[0]);
  if ( v14 != v15 )
    j_j___libc_free_0((unsigned __int64)v14);
  if ( (_QWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0]);
}
