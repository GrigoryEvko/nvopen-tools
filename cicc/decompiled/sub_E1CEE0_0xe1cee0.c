// Function: sub_E1CEE0
// Address: 0xe1cee0
//
__int64 __fastcall sub_E1CEE0(__int64 a1, char a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // r14
  _BYTE *v8; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r12
  _BYTE *v14; // rax
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rdx

  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, &unk_3C1BC40)
    || (unsigned __int8)sub_E0F5E0((const void **)a1, 3u, "__Z") )
  {
    v6 = sub_E1C560((const void **)a1, a2);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = *(_BYTE **)a1;
      if ( *(_QWORD *)a1 == v7 )
        return v6;
      if ( *v8 == 46 )
      {
        v15 = sub_E0E790(a1 + 816, 40, v2, v3, v4, v5);
        if ( v15 )
        {
          *(_QWORD *)(v15 + 16) = v6;
          *(_WORD *)(v15 + 8) = 16385;
          v16 = *(_BYTE *)(v15 + 10);
          *(_QWORD *)(v15 + 24) = v7 - (_QWORD)v8;
          *(_QWORD *)(v15 + 32) = v8;
          *(_BYTE *)(v15 + 10) = v16 & 0xF0 | 5;
          *(_QWORD *)v15 = &unk_49DEDC8;
        }
        v6 = v15;
        *(_QWORD *)a1 = *(_QWORD *)(a1 + 8);
        return v6;
      }
    }
    return 0;
  }
  if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 4u, "___Z")
    && !(unsigned __int8)sub_E0F5E0((const void **)a1, 5u, "____Z") )
  {
    v6 = sub_E1AEA0(a1, 5, v10, v11, v12);
    if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
      return v6;
    return 0;
  }
  v13 = sub_E1C560((const void **)a1, a2);
  if ( !v13 || !(unsigned __int8)sub_E0F5E0((const void **)a1, 0xDu, "_block_invoke") )
    return 0;
  v14 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v14 != 95 )
  {
    sub_E0DEF0((char **)a1, 0);
  }
  else
  {
    *(_QWORD *)a1 = v14 + 1;
    if ( !sub_E0DEF0((char **)a1, 0) )
      return 0;
  }
  v17 = *(_QWORD *)(a1 + 8);
  if ( v17 != *(_QWORD *)a1 )
  {
    if ( **(_BYTE **)a1 != 46 )
      return 0;
    *(_QWORD *)a1 = v17;
  }
  return sub_E0FEB0(a1 + 816, "invocation function for block in ", v13);
}
