// Function: sub_38EF650
// Address: 0x38ef650
//
__int64 __fastcall sub_38EF650(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const char *v7; // rax
  __int64 v9; // rax
  size_t v10; // r13
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r9
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r9
  unsigned __int64 v19; // r8
  char *v20; // rsi
  char v21; // dl
  char v22; // al
  __int64 v23; // [rsp+8h] [rbp-68h]
  unsigned __int64 v24; // [rsp+10h] [rbp-60h]
  unsigned __int64 v25; // [rsp+18h] [rbp-58h]
  const char *v26; // [rsp+20h] [rbp-50h] BYREF
  char v27; // [rsp+30h] [rbp-40h]
  char v28; // [rsp+31h] [rbp-3Fh]

  if ( **(_DWORD **)(a1 + 152) != 3 )
    goto LABEL_2;
  v9 = sub_3909460(a1);
  v10 = 0;
  v11 = *(_QWORD *)(v9 + 16);
  if ( v11 )
  {
    v12 = v11 - 1;
    if ( v11 == 1 )
      v12 = 1;
    if ( v12 > v11 )
      v12 = *(_QWORD *)(v9 + 16);
    v11 = 1;
    v10 = v12 - 1;
  }
  v13 = *(_QWORD *)(v9 + 8);
  sub_38EB180(a1);
  if ( **(_DWORD **)(a1 + 152) != 25 )
  {
    v28 = 1;
    v7 = "expected comma after first string for '.ifeqs' directive";
    if ( !a2 )
      v7 = "expected comma after first string for '.ifnes' directive";
    goto LABEL_4;
  }
  sub_38EB180(a1);
  if ( **(_DWORD **)(a1 + 152) != 3 )
  {
LABEL_2:
    v28 = 1;
    v7 = "expected string parameter for '.ifeqs' directive";
    if ( !a2 )
      v7 = "expected string parameter for '.ifnes' directive";
LABEL_4:
    v26 = v7;
    v27 = 3;
    return sub_3909CF0(a1, &v26, 0, 0, a5, a6);
  }
  v14 = sub_3909460(a1);
  v15 = 0;
  v16 = *(_QWORD *)(v14 + 16);
  if ( v16 )
  {
    v17 = v16 - 1;
    if ( v16 == 1 )
      v17 = 1;
    if ( v17 > v16 )
      v17 = *(_QWORD *)(v14 + 16);
    v16 = 1;
    v15 = v17 - 1;
  }
  v24 = v15;
  v25 = v16;
  v23 = *(_QWORD *)(v14 + 8);
  sub_38EB180(a1);
  v18 = v25;
  v19 = v24;
  v20 = *(char **)(a1 + 400);
  if ( v20 == *(char **)(a1 + 408) )
  {
    sub_38E9AD0((unsigned __int64 *)(a1 + 392), v20, (_QWORD *)(a1 + 380));
    v19 = v24;
    v18 = v25;
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = *(_QWORD *)(a1 + 380);
      v20 = *(char **)(a1 + 400);
    }
    *(_QWORD *)(a1 + 400) = v20 + 8;
  }
  *(_DWORD *)(a1 + 380) = 1;
  v21 = a2;
  v22 = 0;
  if ( v10 == v19 )
  {
    v21 = a2 ^ 1;
    v22 = 1;
    if ( v10 )
    {
      v22 = memcmp((const void *)(v13 + v11), (const void *)(v18 + v23), v10) == 0;
      v21 = a2 ^ v22;
    }
  }
  *(_BYTE *)(a1 + 385) = v21;
  *(_BYTE *)(a1 + 384) = a2 == v22;
  return 0;
}
