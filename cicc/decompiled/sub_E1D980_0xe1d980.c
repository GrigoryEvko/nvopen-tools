// Function: sub_E1D980
// Address: 0xe1d980
//
__int64 __fastcall sub_E1D980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _WORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  bool v9; // cf
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r14
  char *v20; // r13
  __int64 result; // rax
  char v22; // dl
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9

  v6 = *(_WORD **)a1;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = v7 == *(_QWORD *)a1;
  v8 = v7 - *(_QWORD *)a1;
  if ( !v9 && v8 != 1 && *v6 == 29524 )
  {
    v19 = 6;
    *(_QWORD *)a1 = v6 + 1;
    v20 = "struct";
    v16 = sub_E1D370(a1, 0, v8, a4, a5, a6);
    if ( !v16 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Tu") )
    {
      v16 = sub_E1D370(a1, 0, v10, v11, v12, v13);
      if ( v16 )
      {
        v19 = 5;
        v20 = "union";
        goto LABEL_7;
      }
      return 0;
    }
    if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Te") )
      return sub_E1D370(a1, 0, v23, v24, v25, v26);
    v16 = sub_E1D370(a1, 0, v23, v24, v25, v26);
    if ( !v16 )
      return 0;
    v19 = 4;
    v20 = "enum";
  }
LABEL_7:
  result = sub_E0E790(a1 + 816, 40, v14, v15, v17, v18);
  if ( result )
  {
    *(_QWORD *)(result + 16) = v19;
    *(_WORD *)(result + 8) = 16390;
    v22 = *(_BYTE *)(result + 10);
    *(_QWORD *)(result + 24) = v20;
    *(_QWORD *)(result + 32) = v16;
    *(_BYTE *)(result + 10) = v22 & 0xF0 | 5;
    *(_QWORD *)result = &unk_49DF068;
  }
  return result;
}
