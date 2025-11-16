// Function: sub_C64D30
// Address: 0xc64d30
//
void __fastcall __noreturn sub_C64D30(__int64 a1, unsigned __int8 a2)
{
  void (__fastcall *v3)(__int64, _QWORD, _QWORD); // r12
  __int64 v4; // r15
  __int64 v5; // r12
  const void *v6; // rsi
  _QWORD v7[6]; // [rsp+0h] [rbp-D0h] BYREF
  const void **v8; // [rsp+30h] [rbp-A0h]
  _QWORD v9[3]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE v10[120]; // [rsp+58h] [rbp-78h] BYREF

  sub_C64C70(&stru_4F840A0);
  v3 = (void (__fastcall *)(__int64, _QWORD, _QWORD))qword_4F840E0;
  v4 = qword_4F840D8;
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4F840A0);
  if ( v3 )
  {
    sub_CA0F50(v9, a1);
    v3(v4, v9[0], a2);
    sub_2240A30(v9);
  }
  else
  {
    v9[0] = v10;
    v7[5] = 0x100000000LL;
    v7[0] = &unk_49DD288;
    v8 = (const void **)v9;
    v9[1] = 0;
    v9[2] = 64;
    v7[1] = 2;
    memset(&v7[2], 0, 24);
    sub_CB5980(v7, 0, 0, 0);
    v5 = sub_904010((__int64)v7, "LLVM ERROR: ");
    sub_CA0E80(a1, v5);
    sub_904010(v5, "\n");
    v6 = *v8;
    write(2, *v8, (size_t)v8[1]);
    v7[0] = &unk_49DD388;
    sub_CB5840(v7);
    if ( (_BYTE *)v9[0] != v10 )
      _libc_free(v9[0], v6);
  }
  sub_C8C6A0();
  if ( !a2 )
    exit(1);
  abort();
}
