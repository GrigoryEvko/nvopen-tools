// Function: sub_2C361A0
// Address: 0x2c361a0
//
void __fastcall sub_2C361A0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  _QWORD *v8; // r12
  unsigned __int64 v9; // rdi
  int v10; // r14d
  unsigned __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1 + 16;
  v8 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v11, a6);
  sub_2BF6E30(a1, v8);
  v9 = *(_QWORD *)a1;
  v10 = v11[0];
  if ( v9 != v6 )
    _libc_free(v9);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v10;
}
