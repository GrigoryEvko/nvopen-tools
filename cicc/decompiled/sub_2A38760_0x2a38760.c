// Function: sub_2A38760
// Address: 0x2a38760
//
void __fastcall sub_2A38760(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v6[2]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v7; // [rsp-68h] [rbp-68h] BYREF
  __int64 *v8; // [rsp-58h] [rbp-58h]
  __int64 v9; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)a2 == 17 )
  {
    v4 = *(_QWORD **)(a2 + 24);
    if ( *(_DWORD *)(a2 + 32) > 0x40u )
      v4 = (_QWORD *)*v4;
    sub_B18290(a3, " Memory operation size: ", 0x18u);
    sub_B16B10(v6, "StoreSize", 9, (unsigned __int64)v4);
    v5 = sub_2A38130(a3, (__int64)v6);
    sub_B18290(v5, " bytes.", 7u);
    if ( v8 != &v9 )
      j_j___libc_free_0((unsigned __int64)v8);
    if ( (__int64 *)v6[0] != &v7 )
      j_j___libc_free_0(v6[0]);
  }
}
