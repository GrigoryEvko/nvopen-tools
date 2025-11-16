// Function: sub_1650890
// Address: 0x1650890
//
__int64 __fastcall sub_1650890(__int64 *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  _QWORD *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 result; // rax
  __int64 v9; // r9
  _BYTE *v10; // rax
  bool v11; // zf
  __int64 v12; // rdi
  void *v13; // rdx
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r14
  _BYTE *v18; // rax
  __int64 v19; // rdi
  void *v20; // rdx
  __int64 v21; // rax
  _WORD *v22; // rdx
  __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // r15
  __int64 *v27; // rdi
  __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v29; // [rsp+8h] [rbp-58h]
  _QWORD v30[2]; // [rsp+10h] [rbp-50h] BYREF
  char v31; // [rsp+20h] [rbp-40h]
  char v32; // [rsp+21h] [rbp-3Fh]

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 <= 0x17u )
  {
    if ( v2 )
      return 1;
    v23 = (__int64 *)a1[1];
    v24 = *(_QWORD *)(a2 + 40);
    v25 = v23[1];
    if ( v24 != v25 )
    {
      v26 = *a1;
      v27 = (__int64 *)a1[1];
      v32 = 1;
      v31 = 3;
      v30[0] = "Global is used by function in a different module";
      sub_164FF40(v27, (__int64)v30);
      if ( *v23 )
      {
        sub_164FA80(v23, v26);
        sub_164EDD0(*v23, v25);
        sub_164FA80(v23, a2);
        sub_164EDD0(*v23, v24);
        return 0;
      }
    }
    return 0;
  }
  v3 = (_QWORD *)a1[1];
  v4 = *(_QWORD *)(a2 + 40);
  v5 = v3[1];
  if ( v4 )
  {
    v6 = *(_QWORD *)(v4 + 56);
    if ( v6 )
    {
      v7 = *(_QWORD *)(v6 + 40);
      result = 0;
      if ( v7 == v5 )
        return result;
      v32 = 1;
      v9 = *a1;
      v30[0] = "Global is referenced in a different module!";
      v31 = 3;
      if ( *v3 )
      {
        v28 = v9;
        v29 = *v3;
        sub_16E2CE0(v30, *v3);
        v9 = v28;
        v10 = *(_BYTE **)(v29 + 24);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v29 + 16) )
        {
          sub_16E7DE0(v29, 10);
          v9 = v28;
        }
        else
        {
          *(_QWORD *)(v29 + 24) = v10 + 1;
          *v10 = 10;
        }
      }
      v11 = *v3 == 0;
      *((_BYTE *)v3 + 72) = 1;
      if ( !v11 )
      {
        sub_164FA80(v3, v9);
        v12 = *v3;
        v13 = *(void **)(*v3 + 24LL);
        if ( *(_QWORD *)(*v3 + 16LL) - (_QWORD)v13 <= 0xDu )
        {
          v12 = sub_16E7EE0(v12, "; ModuleID = '", 14);
        }
        else
        {
          qmemcpy(v13, "; ModuleID = '", 14);
          *(_QWORD *)(v12 + 24) += 14LL;
        }
        v14 = sub_16E7EE0(v12, *(const char **)(v5 + 176), *(_QWORD *)(v5 + 184));
        v15 = *(_WORD **)(v14 + 24);
        if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
        {
          sub_16E7EE0(v14, "'\n", 2);
        }
        else
        {
          *v15 = 2599;
          *(_QWORD *)(v14 + 24) += 2LL;
        }
        sub_164FA80(v3, a2);
        sub_164FA80(v3, v6);
        sub_164EDD0(*v3, v7);
      }
      return 0;
    }
  }
  v32 = 1;
  v16 = *a1;
  v30[0] = "Global is referenced by parentless instruction!";
  v31 = 3;
  v17 = *v3;
  if ( *v3 )
  {
    sub_16E2CE0(v30, *v3);
    v18 = *(_BYTE **)(v17 + 24);
    if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 16) )
    {
      sub_16E7DE0(v17, 10);
    }
    else
    {
      *(_QWORD *)(v17 + 24) = v18 + 1;
      *v18 = 10;
    }
  }
  v11 = *v3 == 0;
  *((_BYTE *)v3 + 72) = 1;
  if ( v11 )
    return 0;
  sub_164FA80(v3, v16);
  v19 = *v3;
  v20 = *(void **)(*v3 + 24LL);
  if ( *(_QWORD *)(*v3 + 16LL) - (_QWORD)v20 <= 0xDu )
  {
    v19 = sub_16E7EE0(v19, "; ModuleID = '", 14);
  }
  else
  {
    qmemcpy(v20, "; ModuleID = '", 14);
    *(_QWORD *)(v19 + 24) += 14LL;
  }
  v21 = sub_16E7EE0(v19, *(const char **)(v5 + 176), *(_QWORD *)(v5 + 184));
  v22 = *(_WORD **)(v21 + 24);
  if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 1u )
  {
    sub_16E7EE0(v21, "'\n", 2);
  }
  else
  {
    *v22 = 2599;
    *(_QWORD *)(v21 + 24) += 2LL;
  }
  sub_164FA80(v3, a2);
  return 0;
}
