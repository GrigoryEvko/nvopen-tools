// Function: sub_16BCFB0
// Address: 0x16bcfb0
//
void __fastcall __noreturn sub_16BCFB0(__int64 a1, unsigned __int8 a2)
{
  unsigned int v2; // eax
  void (__fastcall *v3)(__int64, unsigned __int64 *, _QWORD); // r13
  __int64 v4; // r14
  __int64 v5; // r13
  _QWORD v6[4]; // [rsp+0h] [rbp-B0h] BYREF
  int v7; // [rsp+20h] [rbp-90h]
  unsigned __int64 *v8; // [rsp+28h] [rbp-88h]
  unsigned __int64 v9[2]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE v10[112]; // [rsp+40h] [rbp-70h] BYREF

  if ( !&_pthread_key_create || (v2 = pthread_mutex_lock(&stru_4FA03A0)) == 0 )
  {
    v3 = (void (__fastcall *)(__int64, unsigned __int64 *, _QWORD))qword_4FA03E0;
    v4 = qword_4FA03D8;
    if ( &_pthread_key_create )
      pthread_mutex_unlock(&stru_4FA03A0);
    if ( v3 )
    {
      sub_16E2FC0(v9, a1);
      v3(v4, v9, a2);
      sub_2240A30(v9);
    }
    else
    {
      v9[0] = (unsigned __int64)v10;
      v9[1] = 0x4000000000LL;
      v8 = v9;
      v6[0] = &unk_49EFC48;
      v7 = 1;
      memset(&v6[1], 0, 24);
      sub_16E7A40(v6, 0, 0, 0);
      v5 = sub_1263B40((__int64)v6, "LLVM ERROR: ");
      sub_16E2CE0(a1, v5);
      sub_1263B40(v5, "\n");
      write(2, (const void *)*v8, *((unsigned int *)v8 + 2));
      v6[0] = &unk_49EFD28;
      sub_16E7960(v6);
      if ( (_BYTE *)v9[0] != v10 )
        _libc_free(v9[0]);
    }
    sub_16CC780();
    exit(1);
  }
  sub_4264C5(v2);
}
