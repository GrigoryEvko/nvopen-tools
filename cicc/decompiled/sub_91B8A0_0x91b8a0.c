// Function: sub_91B8A0
// Address: 0x91b8a0
//
void __fastcall __noreturn sub_91B8A0(char *s, _DWORD *a2, char a3)
{
  const char *v4; // rdi
  char *v5; // r13
  char *v6; // r12

  v4 = "Internal Compiler Error (codegen): ";
  if ( !a3 )
    v4 = byte_3F871B3;
  v5 = strdup(v4);
  v6 = strdup(s);
  sub_686610(0xE52u, a2, (__int64)v5, (__int64)v6);
  _libc_free(v5, a2);
  _libc_free(v6, a2);
  exit(2);
}
