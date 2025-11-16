// Function: sub_23C0270
// Address: 0x23c0270
//
void __fastcall sub_23C0270(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  void (__fastcall *v4)(__int64 *, _QWORD **, const char *, __int64, __int64 *); // r15
  _QWORD *(__fastcall *v5)(_QWORD *, __int64); // rax
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // [rsp+18h] [rbp-58h] BYREF
  __int64 v15[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v16[64]; // [rsp+30h] [rbp-40h] BYREF

  v2 = *a1;
  v3 = *a2;
  v15[0] = (__int64)v16;
  v15[1] = 0;
  v16[0] = 0;
  v4 = *(void (__fastcall **)(__int64 *, _QWORD **, const char *, __int64, __int64 *))(v2 + 24);
  if ( v3 )
  {
    v5 = *(_QWORD *(__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v3 + 16LL);
    if ( v5 == sub_23AEE80 )
    {
      v6 = (_QWORD *)sub_22077B0(0x68u);
      v10 = v6;
      if ( v6 )
      {
        *v6 = &unk_4A16218;
        sub_C8CD80((__int64)(v6 + 1), (__int64)(v6 + 5), v3 + 8, v7, v8, v9);
        sub_C8CD80((__int64)(v10 + 7), (__int64)(v10 + 11), v3 + 56, v11, v12, v13);
      }
      v14 = v10;
    }
    else
    {
      v5(&v14, v3);
    }
  }
  else
  {
    v14 = 0;
  }
  v4(a1, &v14, "Initial IR", 10, v15);
  if ( v14 )
    (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
  sub_23BFF20((__int64)a1, v15, (__int64)"Initial IR", 10);
  if ( (_BYTE *)v15[0] != v16 )
    j_j___libc_free_0(v15[0]);
}
