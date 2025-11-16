// Function: sub_1C26A00
// Address: 0x1c26a00
//
void __fastcall sub_1C26A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char *v6; // r13
  size_t v7; // rax
  int v8; // [rsp-88h] [rbp-88h] BYREF
  __int64 v9; // [rsp-80h] [rbp-80h]
  __int64 v10[15]; // [rsp-78h] [rbp-78h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 208LL);
  if ( v5 )
  {
    v6 = *(char **)(v5 + 88);
    if ( v6 )
    {
      v8 = 0;
      v9 = sub_2241E40(a1, a2, a3, a4, a5);
      v7 = strlen(v6);
      sub_16E8AF0((__int64)v10, v6, v7, (__int64)&v8, 0);
      sub_1C23B90(a1, (__int64)v10, 0, 1);
      if ( v10[3] != v10[1] )
        sub_16E7BA0(v10);
      sub_16E7C30((int *)v10, (__int64)v10);
    }
  }
}
