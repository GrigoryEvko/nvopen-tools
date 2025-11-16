// Function: sub_EB4010
// Address: 0xeb4010
//
__int64 __fastcall sub_EB4010(__int64 a1, char a2)
{
  const char *v3; // rax
  char v5; // r15
  __int64 v6; // rax
  size_t v7; // r13
  char *v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rcx
  char *v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rcx
  const void *v15; // r8
  char *v16; // rsi
  char v17; // al
  char v18; // r15
  void *s2; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  const char *v21; // [rsp+10h] [rbp-60h] BYREF
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]

  if ( **(_DWORD **)(a1 + 48) != 3 )
    goto LABEL_2;
  v5 = a2;
  v6 = sub_ECD7B0(a1);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(char **)(v6 + 8);
  if ( v7 )
  {
    v9 = v7 - 1;
    if ( !v9 )
      v9 = 1;
    ++v8;
    v7 = v9 - 1;
  }
  sub_EABFE0(a1);
  if ( **(_DWORD **)(a1 + 48) != 26 )
  {
    v23 = 1;
    v3 = "expected comma after first string for '.ifeqs' directive";
    if ( !a2 )
      v3 = "expected comma after first string for '.ifnes' directive";
    goto LABEL_4;
  }
  sub_EABFE0(a1);
  if ( **(_DWORD **)(a1 + 48) != 3 )
  {
LABEL_2:
    v23 = 1;
    v3 = "expected string parameter for '.ifeqs' directive";
    if ( !a2 )
      v3 = "expected string parameter for '.ifnes' directive";
LABEL_4:
    v21 = v3;
    v22 = 3;
    return sub_ECE0E0(a1, &v21, 0, 0);
  }
  v10 = sub_ECD7B0(a1);
  v11 = *(_QWORD *)(v10 + 16);
  v12 = *(char **)(v10 + 8);
  if ( v11 )
  {
    v13 = v11 - 1;
    if ( !v13 )
      v13 = 1;
    ++v12;
    v11 = v13 - 1;
  }
  s2 = v12;
  v20 = v11;
  sub_EABFE0(a1);
  v14 = v20;
  v15 = s2;
  v16 = *(char **)(a1 + 328);
  if ( v16 == *(char **)(a1 + 336) )
  {
    sub_EA9230((char **)(a1 + 320), v16, (_QWORD *)(a1 + 308));
    v15 = s2;
    v14 = v20;
  }
  else
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = *(_QWORD *)(a1 + 308);
      v16 = *(char **)(a1 + 328);
    }
    *(_QWORD *)(a1 + 328) = v16 + 8;
  }
  *(_DWORD *)(a1 + 308) = 1;
  if ( v7 == v14 )
  {
    if ( v7 )
    {
      v18 = memcmp(v8, v15, v7) == 0;
      v17 = a2 == v18;
      v5 = a2 ^ v18;
    }
    else
    {
      v17 = a2;
      v5 = a2 ^ 1;
    }
  }
  else
  {
    v17 = a2 ^ 1;
  }
  *(_BYTE *)(a1 + 312) = v17;
  *(_BYTE *)(a1 + 313) = v5;
  return 0;
}
