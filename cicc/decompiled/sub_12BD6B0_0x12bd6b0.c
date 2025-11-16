// Function: sub_12BD6B0
// Address: 0x12bd6b0
//
__int64 __fastcall sub_12BD6B0(__int64 a1)
{
  int v2; // r9d
  const char *v3; // rdi
  size_t v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  char *v7; // rdi
  size_t v8; // rax
  __int64 *v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  char *v13; // rsi
  unsigned int v14; // eax
  char *v15; // r13
  unsigned int v16; // r12d
  __int64 v17; // rdx
  __int64 v18; // rbx
  size_t v19; // rdx
  __int64 v20; // rcx
  __int64 v22; // [rsp-8h] [rbp-58h]
  int v23; // [rsp+Ch] [rbp-44h]
  int v24; // [rsp+Ch] [rbp-44h]
  int v25[2]; // [rsp+10h] [rbp-40h] BYREF
  char *s; // [rsp+18h] [rbp-38h] BYREF

  v2 = **(unsigned __int8 **)(a1 + 16);
  v3 = **(const char ***)(a1 + 8);
  v4 = 0;
  v5 = (__int64)v3;
  if ( v3 )
  {
    v23 = v2;
    v4 = strlen(v3);
    v2 = v23;
  }
  v6 = v4;
  v7 = **(char ***)a1;
  v8 = 0;
  if ( v7 )
  {
    v24 = v2;
    v8 = strlen(v7);
    v2 = v24;
  }
  sub_16D8B50((int)v25, (int)"LTO", 3, (int)"LTO step.", 9, v2, v7, v8, v5, v6);
  v9 = *(__int64 **)(a1 + 48);
  v10 = *(_QWORD **)(a1 + 32);
  s = 0;
  v11 = *v9;
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(char **)(v12 + 8);
  v14 = sub_12F5F30(*(_DWORD *)v12, (_DWORD)v13, *v10, *(_QWORD *)(a1 + 40), (int)v11 + 48, (unsigned int)&s, v11 + 208);
  v15 = s;
  v16 = v14;
  v17 = v22;
  if ( s )
  {
    v18 = **(_QWORD **)(a1 + 48);
    v19 = strlen(s);
    if ( v19 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(v18 + 88) )
      sub_4262D8((__int64)"basic_string::append");
    v13 = v15;
    sub_2241490(v18 + 80, v15, v19, v20);
    if ( s )
      j_j___libc_free_0_0(s);
    s = 0;
  }
  if ( *(_QWORD *)v25 )
    sub_16D7950(*(_QWORD *)v25, v13, v17);
  return v16;
}
