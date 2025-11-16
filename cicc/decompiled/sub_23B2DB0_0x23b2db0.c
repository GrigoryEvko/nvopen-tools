// Function: sub_23B2DB0
// Address: 0x23b2db0
//
void __fastcall sub_23B2DB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r12
  _QWORD *(__fastcall *v5)(_QWORD *, __int64); // rax
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // [rsp+18h] [rbp-38h] BYREF

  v4 = *a4;
  *a4 = 0;
  if ( v4 )
  {
    v5 = *(_QWORD *(__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v4 + 16LL);
    if ( v5 == sub_23AEE80 )
    {
      v6 = (_QWORD *)sub_22077B0(0x68u);
      v10 = v6;
      if ( v6 )
      {
        *v6 = &unk_4A16218;
        sub_C8CD80((__int64)(v6 + 1), (__int64)(v6 + 5), v4 + 8, v7, v8, v9);
        sub_C8CD80((__int64)(v10 + 7), (__int64)(v10 + 11), v4 + 56, v11, v12, v13);
      }
      v14 = v10;
    }
    else
    {
      v5(&v14, v4);
    }
    nullsub_1493();
    if ( v14 )
      (*(void (__fastcall **)(_QWORD *, __int64))(*v14 + 8LL))(v14, a2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 8LL))(v4, a2);
  }
  else
  {
    nullsub_1493();
  }
}
