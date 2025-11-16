// Function: sub_E1FF00
// Address: 0xe1ff00
//
__int64 __fastcall sub_E1FF00(char **a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9

  if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "dn") )
  {
    if ( a1[1] == *a1 || (unsigned int)(**a1 - 48) > 9 )
      v9 = sub_E18B30(a1, 2, v1, v2, v3, v4);
    else
      v9 = sub_E1FE40((__int64 *)a1);
    if ( v9 )
    {
      v10 = sub_E0E790((__int64)(a1 + 102), 24, v5, v6, v7, v8);
      v11 = v10;
      if ( v10 )
      {
        *(_QWORD *)(v10 + 16) = v9;
        *(_WORD *)(v10 + 8) = 16434;
        *(_BYTE *)(v10 + 10) = *(_BYTE *)(v10 + 10) & 0xF0 | 5;
        *(_QWORD *)v10 = &unk_49E0088;
      }
      return v11;
    }
  }
  else
  {
    sub_E0F5E0((const void **)a1, 2u, "on");
    v11 = sub_E1C280((__int64)a1, 0);
    if ( v11 )
    {
      if ( a1[1] == *a1 || **a1 != 73 )
        return v11;
      v17 = sub_E1F700((__int64)a1, 0, v13, v14, v15, v16);
      if ( v17 )
        return sub_E0FC10((__int64)(a1 + 102), v11, v17, v18, v19, v20);
    }
  }
  return 0;
}
