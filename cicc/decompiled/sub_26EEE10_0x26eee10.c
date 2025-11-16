// Function: sub_26EEE10
// Address: 0x26eee10
//
__int64 __fastcall sub_26EEE10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rsi
  _QWORD *v7; // r13
  unsigned __int64 v8; // rdi
  _QWORD *v9; // r13
  unsigned __int64 v10; // rdi
  __int64 *v11; // rdi
  __int64 i; // [rsp+18h] [rbp-108h]
  _QWORD v14[2]; // [rsp+20h] [rbp-100h] BYREF
  __int64 v15; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v16; // [rsp+40h] [rbp-E0h] BYREF
  __int64 *v17; // [rsp+48h] [rbp-D8h]
  __int64 v18; // [rsp+58h] [rbp-C8h] BYREF
  void *v19; // [rsp+70h] [rbp-B0h]
  __int64 v20; // [rsp+78h] [rbp-A8h]
  _QWORD *v21; // [rsp+80h] [rbp-A0h]
  __int64 v22; // [rsp+88h] [rbp-98h]
  char v23; // [rsp+A0h] [rbp-80h] BYREF
  void *s; // [rsp+A8h] [rbp-78h]
  __int64 v25; // [rsp+B0h] [rbp-70h]
  _QWORD *v26; // [rsp+B8h] [rbp-68h]
  __int64 v27; // [rsp+C0h] [rbp-60h]
  char v28; // [rsp+D8h] [rbp-48h] BYREF

  sub_2A3FB20(v14, a3);
  sub_BA8E40(a3, "llvm.pseudo_probe_desc", 0x16u);
  v5 = *(_QWORD *)(a3 + 32);
  for ( i = a3 + 24; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v6 = v5 - 56;
    if ( !v5 )
      v6 = 0;
    if ( !sub_B2FC80(v6) )
    {
      sub_26EEC90((__int64)&v16, v6, (__int64)v14);
      sub_26EA7B0(&v16, v6);
      v7 = v26;
      while ( v7 )
      {
        v8 = (unsigned __int64)v7;
        v7 = (_QWORD *)*v7;
        j_j___libc_free_0(v8);
      }
      memset(s, 0, 8 * v25);
      v27 = 0;
      v26 = 0;
      if ( s != &v28 )
        j_j___libc_free_0((unsigned __int64)s);
      v9 = v21;
      while ( v9 )
      {
        v10 = (unsigned __int64)v9;
        v9 = (_QWORD *)*v9;
        j_j___libc_free_0(v10);
      }
      memset(v19, 0, 8 * v20);
      v22 = 0;
      v21 = 0;
      if ( v19 != &v23 )
        j_j___libc_free_0((unsigned __int64)v19);
      if ( v17 != &v18 )
        j_j___libc_free_0((unsigned __int64)v17);
    }
  }
  memset((void *)a1, 0, 0x60u);
  v11 = (__int64 *)v14[0];
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_BYTE *)(a1 + 28) = 1;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  if ( v11 != &v15 )
    j_j___libc_free_0((unsigned __int64)v11);
  return a1;
}
