// Function: sub_E1F480
// Address: 0xe1f480
//
__int64 __fastcall sub_E1F480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rdx
  char *v7; // rax
  __int64 result; // rax
  char *v9; // rdx
  char *v10; // rax
  char *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  void *v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  char v24; // dl
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r13
  char v31; // dl
  __int64 v32[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(char **)(a1 + 8);
  v7 = *(char **)a1;
  if ( *(char **)a1 == v6 )
    return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
  a4 = (unsigned __int8)*v7;
  a2 = v6 - v7;
  if ( (_BYTE)a4 == 84 )
  {
    if ( a2 == 1 )
      return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
    a2 = (unsigned int)v7[1];
    v10 = (char *)memchr("yptnk", a2, 5u);
    if ( !v10 )
      return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
    v6 = "";
    if ( v10 == "" )
      return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
    v25 = sub_E1DAD0(a1, 0);
    if ( v25 )
    {
      v30 = sub_E1F480(a1);
      if ( v30 )
      {
        result = sub_E0E790(a1 + 816, 32, v26, v27, v28, v29);
        if ( result )
        {
          *(_QWORD *)(result + 16) = v25;
          *(_WORD *)(result + 8) = 16418;
          v31 = *(_BYTE *)(result + 10);
          *(_QWORD *)(result + 24) = v30;
          *(_BYTE *)(result + 10) = v31 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49DFA88;
        }
        return result;
      }
    }
    return 0;
  }
  if ( (char)a4 > 84 )
  {
    if ( (_BYTE)a4 != 88 )
      return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
    *(_QWORD *)a1 = v7 + 1;
    result = sub_E18BB0(a1);
    if ( result )
    {
      v9 = *(char **)a1;
      if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
        goto LABEL_12;
    }
    return 0;
  }
  if ( (_BYTE)a4 != 74 )
  {
    if ( (_BYTE)a4 == 76 )
    {
      if ( a2 == 1 || v7[1] != 90 )
        return sub_E1ED50(a1, a2, (__int64)v6, a4, a5, a6);
      *(_QWORD *)a1 = v7 + 2;
      result = sub_E1C560((const void **)a1, 1);
      if ( !result )
        return 0;
      v9 = *(char **)a1;
      if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
        return 0;
LABEL_12:
      if ( *v9 == 69 )
      {
        *(_QWORD *)a1 = v9 + 1;
        return result;
      }
      return 0;
    }
    return sub_E1AEA0(a1, a2, (__int64)v6, a4, a5);
  }
  v11 = v7 + 1;
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = v11;
  if ( v11 == v6 )
    goto LABEL_20;
LABEL_19:
  if ( *v11 == 69 )
  {
    *(_QWORD *)a1 = v11 + 1;
    v18 = sub_E11E80((_QWORD *)a1, (v12 - v13) >> 3, (__int64)v6, a4, a5, a6);
    v20 = v19;
    result = sub_E0E790(a1 + 816, 32, v19, v21, v22, v23);
    if ( result )
    {
      v24 = *(_BYTE *)(result + 10);
      *(_QWORD *)(result + 16) = v18;
      *(_WORD *)(result + 8) = 16425;
      *(_QWORD *)(result + 24) = v20;
      *(_BYTE *)(result + 10) = v24 & 0xF0 | 5;
      *(_QWORD *)result = &unk_49DFD28;
    }
  }
  else
  {
LABEL_20:
    while ( 1 )
    {
      result = sub_E1F480(a1);
      v32[0] = result;
      if ( !result )
        break;
      sub_E18380(a1 + 16, v32, v14, v15, v16, v17);
      v11 = *(char **)a1;
      v6 = *(char **)(a1 + 8);
      if ( *(char **)a1 != v6 )
        goto LABEL_19;
    }
  }
  return result;
}
