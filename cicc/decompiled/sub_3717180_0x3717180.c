// Function: sub_3717180
// Address: 0x3717180
//
char *__fastcall sub_3717180(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  char *v5; // r14
  char *v6; // rdx
  char *v7; // r13
  char **v8; // rax
  const void *v9[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v10; // [rsp+20h] [rbp-30h] BYREF

  LODWORD(v9[0]) = 0;
  if ( a2 <= 0xFFF )
    return sub_370C330(a2);
  v2 = 16LL * ((a2 & 0x7FFFFFFF) - 4096);
  v3 = v2 + *(_QWORD *)(a1 + 112);
  if ( !*(_QWORD *)v3 )
  {
    sub_37FAEF0(v9, a1, a2);
    v5 = sub_C948A0((char ***)(a1 + 104), v9[0], (size_t)v9[1]);
    v7 = v6;
    if ( v9[0] != &v10 )
      j_j___libc_free_0((unsigned __int64)v9[0]);
    v8 = (char **)(v2 + *(_QWORD *)(a1 + 112));
    *v8 = v5;
    v8[1] = v7;
    v3 = v2 + *(_QWORD *)(a1 + 112);
  }
  return *(char **)v3;
}
