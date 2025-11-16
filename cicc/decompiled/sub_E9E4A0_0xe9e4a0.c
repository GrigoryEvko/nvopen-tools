// Function: sub_E9E4A0
// Address: 0xe9e4a0
//
__int64 __fastcall sub_E9E4A0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD v13[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v14; // [rsp+20h] [rbp-60h]
  unsigned int *v15[2]; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+40h] [rbp-40h]
  __int16 v17; // [rsp+50h] [rbp-30h]

  v3 = *(_QWORD *)(a1 + 8);
  LODWORD(v13[0]) = a2;
  v4 = *(_QWORD *)(v3 + 1744);
  v5 = v3 + 1736;
  if ( v4 )
  {
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v4 + 16);
        v7 = *(_QWORD *)(v4 + 24);
        if ( a2 <= *(_DWORD *)(v4 + 32) )
          break;
        v4 = *(_QWORD *)(v4 + 24);
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 16);
    }
    while ( v6 );
LABEL_6:
    if ( v3 + 1736 != v5 && a2 >= *(_DWORD *)(v5 + 32) )
    {
      result = *(_QWORD *)(v5 + 40);
      if ( result )
        return result;
LABEL_11:
      v9 = *(_QWORD *)(a1 + 8);
      v10 = *(_QWORD *)(v9 + 152);
      v11 = *(_QWORD *)(v10 + 88);
      v12 = *(_QWORD *)(v10 + 96);
      v17 = 2306;
      v16 = a2;
      v13[1] = v12;
      v13[2] = "line_table_start";
      v14 = 773;
      v13[0] = v11;
      v15[0] = (unsigned int *)v13;
      result = sub_E6C460(v9, (const char **)v15);
      *(_QWORD *)(v5 + 40) = result;
      return result;
    }
  }
  v15[0] = (unsigned int *)v13;
  v5 = sub_E9E2A0((_QWORD *)(v3 + 1728), v5, v15);
  result = *(_QWORD *)(v5 + 40);
  if ( !result )
    goto LABEL_11;
  return result;
}
