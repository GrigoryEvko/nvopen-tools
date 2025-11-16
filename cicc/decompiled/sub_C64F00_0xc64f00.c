// Function: sub_C64F00
// Address: 0xc64f00
//
void __fastcall __noreturn sub_C64F00(const char *buf, unsigned __int8 a2)
{
  void (__fastcall *v2)(__int64, const char *, _QWORD); // r13
  __int64 v3; // r14
  size_t v4; // rax

  sub_C64C70(&stru_4F84060);
  v2 = (void (__fastcall *)(__int64, const char *, _QWORD))qword_4F840D0;
  v3 = qword_4F840C8;
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4F84060);
  if ( v2 )
  {
    v2(v3, buf, a2);
    BUG();
  }
  write(2, "LLVM ERROR: out of memory\n", 0x1Au);
  v4 = strlen(buf);
  write(2, buf, v4);
  write(2, "\n", 1u);
  abort();
}
