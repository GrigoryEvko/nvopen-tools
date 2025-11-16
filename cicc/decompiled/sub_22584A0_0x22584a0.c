// Function: sub_22584A0
// Address: 0x22584a0
//
__int64 __fastcall sub_22584A0(__int64 a1)
{
  char v2; // r9
  const char *v3; // rdi
  size_t v4; // rax
  char *v5; // r12
  __int64 v6; // r13
  char *v7; // rdi
  size_t v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  unsigned int v11; // eax
  char *v12; // r13
  unsigned int v13; // r12d
  __int64 v14; // rbx
  unsigned __int64 v15; // rdx
  char v17; // [rsp+Ch] [rbp-44h]
  char v18; // [rsp+Ch] [rbp-44h]
  int v19[2]; // [rsp+10h] [rbp-40h] BYREF
  char *s; // [rsp+18h] [rbp-38h] BYREF

  v2 = **(_BYTE **)(a1 + 16);
  v3 = **(const char ***)(a1 + 8);
  v4 = 0;
  v5 = (char *)v3;
  if ( v3 )
  {
    v17 = v2;
    v4 = strlen(v3);
    v2 = v17;
  }
  v6 = v4;
  v7 = **(char ***)a1;
  v8 = 0;
  if ( v7 )
  {
    v18 = v2;
    v8 = strlen(v7);
    v2 = v18;
  }
  sub_CA08F0((__int64 *)v19, "LTO", 3u, (__int64)"LTO step.", 9, v2, v7, v8, v5, v6);
  v9 = *(_QWORD **)(a1 + 48);
  v10 = *(_QWORD **)(a1 + 32);
  s = 0;
  v11 = sub_309A610(
          **(_DWORD **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          *v10,
          *(_QWORD *)(a1 + 40),
          (unsigned int)*v9 + 48,
          (unsigned int)&s,
          *v9 + 208LL);
  v12 = s;
  v13 = v11;
  if ( s )
  {
    v14 = **(_QWORD **)(a1 + 48);
    v15 = strlen(s);
    if ( v15 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(v14 + 88) )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)(v14 + 80), v12, v15);
    if ( s )
      j_j___libc_free_0_0((unsigned __int64)s);
    s = 0;
  }
  if ( *(_QWORD *)v19 )
    sub_C9E2A0(*(__int64 *)v19);
  return v13;
}
