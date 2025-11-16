// Function: sub_3146000
// Address: 0x3146000
//
__int64 __fastcall sub_3146000(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  unsigned __int64 v4; // rdx
  char **v5; // rbx
  char *v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  size_t v9; // rdx
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  char *v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int8 *v18; // r13
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  const char *v21; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v22; // [rsp+8h] [rbp-48h]
  char *v23; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  if ( !sub_B2FC80(a1) )
  {
    v3 = sub_BD5D20(a1);
    if ( v4 > 3
      && *(_DWORD *)v3 == 1920234286
      && (v18 = *(unsigned __int8 **)(a1 - 32), (unsigned int)*v18 - 15 <= 1)
      && (a2 = 8, (unsigned __int8)sub_AC5570(*(_QWORD *)(a1 - 32), 8u)) )
    {
      v16 = sub_AC52D0((__int64)v18);
    }
    else
    {
      if ( (*(_BYTE *)(a1 + 35) & 4) != 0 )
      {
        v5 = (char **)&off_49D8C20;
        v6 = "__cfstring";
        v7 = sub_B31D10(a1, a2, v4);
        v24 = v8;
        v9 = 0;
        v23 = (char *)v7;
        if ( "__DATA,__cfstring" == (char *)-7LL )
          goto LABEL_8;
LABEL_7:
        v9 = strlen(v6);
LABEL_8:
        while ( sub_C931B0((__int64 *)&v23, v6, v9, 0) == -1 )
        {
          if ( &unk_49D8C48 == (_UNKNOWN *)++v5 )
            goto LABEL_16;
          v6 = *v5;
          v9 = 0;
          if ( *v5 )
            goto LABEL_7;
        }
        return sub_3146240(*(_QWORD *)(a1 - 32));
      }
LABEL_16:
      if ( (*(_BYTE *)(a1 + 7) & 0x10) == 0 )
        return 0;
      v16 = (__int64)sub_BD5D20(a1);
    }
    return sub_3145F20(v16, v17);
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x10) == 0 )
    return 0;
  v10 = sub_BD5D20(a1);
  v22 = v11;
  v21 = v10;
  v12 = sub_C93460((__int64 *)&v21, ".content.", 9u);
  if ( v12 == -1 || (v13 = v12 + 9, v13 > v22) || (v14 = (char *)&v21[v13], v15 = v22 - v13, v22 == v13) )
  {
    v19 = sub_C93460((__int64 *)&v21, ".llvm.", 6u);
    if ( v19 == -1 )
    {
      v19 = v22;
    }
    else if ( v22 <= v19 )
    {
      v19 = v22;
    }
    v23 = (char *)v21;
    v24 = v19;
    v20 = sub_C93460((__int64 *)&v23, ".__uniq.", 8u);
    v14 = v23;
    v15 = v20;
    if ( v20 == -1 )
    {
      v15 = v24;
    }
    else if ( v24 <= v20 )
    {
      v15 = v24;
    }
  }
  return sub_CBF760(v14, v15);
}
