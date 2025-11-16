// Function: sub_1C30F40
// Address: 0x1c30f40
//
__int64 __fastcall sub_1C30F40(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  char *v9; // rsi
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  const char *v15; // rdx
  _QWORD v16[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-70h] BYREF
  const char *v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+38h] [rbp-48h]
  int v22; // [rsp+40h] [rbp-40h]
  const char **v23; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v4 = sub_15C70A0(a2);
  if ( !v4 )
    return a1;
  v5 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
  if ( *(_BYTE *)v5 != 15 )
  {
    v5 = *(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8));
    if ( !v5 )
    {
      v15 = byte_3F871B3;
      v9 = (char *)byte_3F871B3;
LABEL_18:
      v18 = (const char *)&v20;
      sub_1C30A70((__int64 *)&v18, v9, (__int64)v15);
      sub_2241490(a1, v18, v19);
      goto LABEL_7;
    }
  }
  v6 = *(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8));
  if ( v6 )
  {
    v7 = sub_161E970(v6);
    v9 = (char *)v7;
    if ( v7 )
    {
      v15 = (const char *)(v7 + v8);
      goto LABEL_18;
    }
  }
  LOBYTE(v20) = 0;
  v18 = (const char *)&v20;
  v19 = 0;
  sub_2241490(a1, (const char *)&v20, 0);
LABEL_7:
  if ( v18 != (const char *)&v20 )
    j_j___libc_free_0(v18, v20 + 1);
  LOBYTE(v17[0]) = 0;
  v16[0] = v17;
  v16[1] = 0;
  v22 = 1;
  v18 = (const char *)&unk_49EFBE0;
  v21 = 0;
  v20 = 0;
  v19 = 0;
  v23 = (const char **)v16;
  v10 = sub_16E7EE0((__int64)&v18, "(", 1u);
  v11 = sub_15C70B0(a2);
  v12 = sub_16E7A90(v10, v11);
  v13 = *(_BYTE **)(v12 + 24);
  if ( *(_BYTE **)(v12 + 16) == v13 )
  {
    sub_16E7EE0(v12, ")", 1u);
  }
  else
  {
    *v13 = 41;
    ++*(_QWORD *)(v12 + 24);
  }
  if ( v21 != v19 )
    sub_16E7BA0((__int64 *)&v18);
  sub_2241490(a1, *v23, v23[1]);
  sub_16E7BC0((__int64 *)&v18);
  if ( (_QWORD *)v16[0] != v17 )
    j_j___libc_free_0(v16[0], v17[0] + 1LL);
  return a1;
}
