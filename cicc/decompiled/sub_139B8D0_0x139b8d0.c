// Function: sub_139B8D0
// Address: 0x139b8d0
//
__int64 __fastcall sub_139B8D0(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v7; // [rsp-B0h] [rbp-B0h] BYREF
  __int64 v8; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v9; // [rsp-98h] [rbp-98h]
  __int64 *v10; // [rsp-88h] [rbp-88h] BYREF
  __int16 v11; // [rsp-78h] [rbp-78h]
  __int64 v12[2]; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v13[2]; // [rsp-58h] [rbp-58h] BYREF
  __m128i v14; // [rsp-48h] [rbp-48h] BYREF
  _QWORD v15[7]; // [rsp-38h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F98A8D )
  {
    v2 += 16;
    if ( v3 == v2 )
      BUG();
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F98A8D);
  v12[0] = (__int64)v13;
  v7 = *(_QWORD *)(v4 + 160);
  sub_1399600(v12, "Call graph", (__int64)"");
  v9 = 260;
  v11 = 260;
  v10 = v12;
  v8 = a1 + 160;
  sub_139B4F0(&v14, (__int64)&v7, (__int64)&v8, 1, (__int64)&v10);
  if ( v14.m128i_i64[1] )
  {
    sub_16BED90(v14.m128i_i64[0], v14.m128i_i64[1], 0, 0);
    v5 = v14.m128i_i64[0];
    if ( (_QWORD *)v14.m128i_i64[0] == v15 )
      goto LABEL_8;
    goto LABEL_7;
  }
  v5 = v14.m128i_i64[0];
  if ( (_QWORD *)v14.m128i_i64[0] != v15 )
LABEL_7:
    j_j___libc_free_0(v5, v15[0] + 1LL);
LABEL_8:
  if ( (_QWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0], v13[0] + 1LL);
  return 0;
}
