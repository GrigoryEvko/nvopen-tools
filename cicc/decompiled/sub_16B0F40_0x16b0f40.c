// Function: sub_16B0F40
// Address: 0x16b0f40
//
void __fastcall __noreturn sub_16B0F40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  char *v6; // rdx
  __int64 v7; // r12
  void *v8; // rdi
  const char *v9; // rsi
  size_t v10; // r13
  char *v11; // rdx

  v4 = sub_16E8CB0(a1, a2, a3);
  v5 = sub_16E7EE0(v4, *(const char **)a1, *(_QWORD *)(a1 + 8));
  v6 = *(char **)(v5 + 24);
  v7 = v5;
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x1Cu )
  {
    v7 = sub_16E7EE0(v5, ": CommandLine Error: Option '", 29);
  }
  else
  {
    qmemcpy(v6, ": CommandLine Error: Option '", 0x1Du);
    *(_QWORD *)(v5 + 24) += 29LL;
  }
  v8 = *(void **)(v7 + 24);
  v9 = *(const char **)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 32);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 < v10 )
  {
    v7 = sub_16E7EE0(v7, v9, v10);
  }
  else if ( v10 )
  {
    memcpy(v8, v9, v10);
    *(_QWORD *)(v7 + 24) += v10;
  }
  v11 = *(char **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v11 <= 0x1Cu )
  {
    sub_16E7EE0(v7, "' registered more than once!\n", 29);
  }
  else
  {
    qmemcpy(v11, "' registered more than once!\n", 0x1Du);
    *(_QWORD *)(v7 + 24) += 29LL;
  }
  sub_16BD130("inconsistency in registered CommandLine options", 1);
}
