// Function: sub_CE0220
// Address: 0xce0220
//
void __fastcall sub_CE0220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char *v6; // r13
  size_t v7; // rax
  int v8; // [rsp-98h] [rbp-98h] BYREF
  __int64 v9; // [rsp-90h] [rbp-90h]
  __int64 v10[17]; // [rsp-88h] [rbp-88h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 208LL);
  if ( v5 )
  {
    v6 = *(char **)(v5 + 88);
    if ( v6 )
    {
      v8 = 0;
      v9 = sub_2241E40(a1, a2, a3, a4, a5);
      v7 = strlen(v6);
      sub_CB7060((__int64)v10, v6, v7, (__int64)&v8, 0);
      sub_CDD2D0(a1, (__int64)v10, 0, 1);
      if ( v10[4] != v10[2] )
        sub_CB5AE0(v10);
      sub_CB5B00((int *)v10, (__int64)v10);
    }
  }
}
