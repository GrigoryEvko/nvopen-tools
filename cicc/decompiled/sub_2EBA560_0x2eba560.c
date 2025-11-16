// Function: sub_2EBA560
// Address: 0x2eba560
//
_BOOL8 __fastcall sub_2EBA560(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  bool v6; // r13
  _BYTE *v7; // rbx
  _BYTE *v8; // r12
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  void *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  void *v15; // rax
  void *v16; // rax
  void *v17; // rax
  __int64 *v18; // rdi
  bool v19; // [rsp+Fh] [rbp-D1h]
  unsigned __int64 v20[2]; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v21[32]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE *v22; // [rsp+40h] [rbp-A0h]
  __int64 v23; // [rsp+48h] [rbp-98h]
  _BYTE v24[56]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v25; // [rsp+88h] [rbp-58h]
  __int64 v26; // [rsp+90h] [rbp-50h]
  char v27; // [rsp+98h] [rbp-48h]
  __int64 v28; // [rsp+9Ch] [rbp-44h]

  v20[1] = 0x400000000LL;
  v23 = 0x600000000LL;
  v1 = *(_QWORD *)(a1 + 128);
  v28 = 0;
  v25 = 0;
  v27 = 0;
  v26 = v1;
  LODWORD(v1) = *(_DWORD *)(v1 + 120);
  v20[0] = (unsigned __int64)v21;
  HIDWORD(v28) = v1;
  v22 = v24;
  sub_2EBA550((__int64)v20);
  v6 = sub_2EB38F0(a1, (__int64)v20, v2, v3, v4, v5);
  if ( v6 )
  {
    v12 = sub_CB72A0();
    v13 = sub_904010((__int64)v12, "Post");
    v14 = sub_904010(v13, "DominatorTree is different than a freshly computed one!\n");
    sub_904010(v14, "\tCurrent:\n");
    v15 = sub_CB72A0();
    sub_2EB4190(a1, (__int64)v15);
    v16 = sub_CB72A0();
    sub_904010((__int64)v16, "\n\tFreshly computed tree:\n");
    v17 = sub_CB72A0();
    sub_2EB4190((__int64)v20, (__int64)v17);
    v18 = (__int64 *)sub_CB72A0();
    if ( v18[4] != v18[2] )
      sub_CB5AE0(v18);
  }
  v7 = v22;
  v19 = !v6;
  v8 = &v22[8 * (unsigned int)v23];
  if ( v22 != v8 )
  {
    do
    {
      v9 = *((_QWORD *)v8 - 1);
      v8 -= 8;
      if ( v9 )
      {
        v10 = *(_QWORD *)(v9 + 24);
        if ( v10 != v9 + 40 )
          _libc_free(v10);
        j_j___libc_free_0(v9);
      }
    }
    while ( v7 != v8 );
    v8 = v22;
  }
  if ( v8 != v24 )
    _libc_free((unsigned __int64)v8);
  if ( (_BYTE *)v20[0] != v21 )
    _libc_free(v20[0]);
  return v19;
}
