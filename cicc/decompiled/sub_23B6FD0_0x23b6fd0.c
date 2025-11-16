// Function: sub_23B6FD0
// Address: 0x23b6fd0
//
void *__fastcall sub_23B6FD0(__m128i a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  _QWORD *(__fastcall *v7)(__int64 *, __int64); // rax
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  bool v16; // bl
  __int64 v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v20[14]; // [rsp+20h] [rbp-70h] BYREF

  v20[5] = 0x100000000LL;
  v20[6] = a6;
  memset(&v20[1], 0, 32);
  v20[0] = &unk_49DD210;
  sub_CB5980((__int64)v20, 0, 0, 0);
  sub_23B2720(&v18, a3);
  v6 = v18;
  if ( v18 )
  {
    v7 = *(_QWORD *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)v18 + 16LL);
    if ( v7 == sub_23AEE80 )
    {
      v8 = (_QWORD *)sub_22077B0(0x68u);
      v12 = (__int64)v8;
      if ( v8 )
      {
        *v8 = &unk_4A16218;
        sub_C8CD80((__int64)(v8 + 1), (__int64)(v8 + 5), v6 + 8, v9, v10, v11);
        sub_C8CD80(v12 + 56, v12 + 88, v6 + 56, v13, v14, v15);
      }
      v19 = v12;
    }
    else
    {
      v7(&v19, v18);
    }
  }
  else
  {
    v19 = 0;
  }
  v16 = sub_23B44D0(&v19);
  if ( v19 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
  if ( v16 )
    sub_23B6C50((char *)v20, &v18, a1);
  if ( v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  v20[0] = &unk_49DD210;
  return sub_CB5840((__int64)v20);
}
