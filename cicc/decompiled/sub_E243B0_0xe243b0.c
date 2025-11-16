// Function: sub_E243B0
// Address: 0xe243b0
//
int __fastcall sub_E243B0(__int64 a1)
{
  __int64 v2; // rsi
  bool v3; // zf
  unsigned __int64 v4; // rbx
  __int64 v5; // rdi
  const char *v6; // rdi
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  int v10; // esi
  const char *v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+18h] [rbp-38h]
  int v16; // [rsp+20h] [rbp-30h]

  v2 = *(unsigned int *)(a1 + 104);
  printf("%d function parameter backreferences\n", v2);
  v3 = *(_QWORD *)(a1 + 104) == 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = -1;
  v16 = 1;
  if ( v3 )
  {
    v6 = 0;
  }
  else
  {
    v4 = 0;
    do
    {
      v5 = *(_QWORD *)(a1 + 8 * v4 + 24);
      v13 = 0;
      (*(void (__fastcall **)(__int64, const char **, _QWORD))(*(_QWORD *)v5 + 16LL))(v5, &v12, 0);
      v2 = (unsigned int)v4++;
      printf("  [%d] - %.*s\n", v2, v13, v12);
    }
    while ( *(_QWORD *)(a1 + 104) > v4 );
    v6 = v12;
  }
  _libc_free(v6, v2);
  if ( *(_QWORD *)(a1 + 104) )
    putchar(10);
  v7 = 0;
  LODWORD(v8) = printf("%d name backreferences\n", *(_DWORD *)(a1 + 192));
  if ( *(_QWORD *)(a1 + 192) )
  {
    do
    {
      v9 = *(_QWORD *)(a1 + 8 * v7 + 112);
      v10 = v7++;
      printf("  [%d] - %.*s\n", v10, *(_QWORD *)(v9 + 24), *(const char **)(v9 + 32));
      v8 = *(_QWORD *)(a1 + 192);
    }
    while ( v8 > v7 );
    if ( v8 )
      LODWORD(v8) = putchar(10);
  }
  return v8;
}
