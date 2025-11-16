// Function: sub_3145E30
// Address: 0x3145e30
//
__int64 __fastcall sub_3145E30(__int64 a1)
{
  __int64 v1; // rbp
  const char *v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  const char *v11; // [rsp-38h] [rbp-38h] BYREF
  unsigned __int64 v12; // [rsp-30h] [rbp-30h]
  const char *v13; // [rsp-28h] [rbp-28h] BYREF
  unsigned __int64 v14; // [rsp-20h] [rbp-20h]
  __int64 v15; // [rsp-8h] [rbp-8h]

  if ( (*(_BYTE *)(a1 + 7) & 0x10) == 0 )
    return 0;
  v15 = v1;
  v3 = sub_BD5D20(a1);
  v12 = v4;
  v11 = v3;
  v5 = sub_C93460((__int64 *)&v11, ".content.", 9u);
  if ( v5 == -1 || (v6 = v5 + 9, v6 > v12) || (v7 = (char *)&v11[v6], v8 = v12 - v6, v12 == v6) )
  {
    v9 = sub_C93460((__int64 *)&v11, ".llvm.", 6u);
    if ( v9 == -1 )
    {
      v9 = v12;
    }
    else if ( v12 <= v9 )
    {
      v9 = v12;
    }
    v13 = v11;
    v14 = v9;
    v10 = sub_C93460((__int64 *)&v13, ".__uniq.", 8u);
    v7 = (char *)v13;
    v8 = v10;
    if ( v10 == -1 )
    {
      v8 = v14;
    }
    else if ( v14 <= v10 )
    {
      v8 = v14;
    }
  }
  return sub_CBF760(v7, v8);
}
