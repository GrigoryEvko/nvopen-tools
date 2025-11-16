// Function: sub_38E1750
// Address: 0x38e1750
//
__int64 __fastcall sub_38E1750(__int64 a1, unsigned int a2)
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
  _QWORD v13[2]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v15; // [rsp+20h] [rbp-70h]
  unsigned int *v16; // [rsp+30h] [rbp-60h] BYREF
  unsigned int *v17[2]; // [rsp+50h] [rbp-40h] BYREF
  __int16 v18; // [rsp+60h] [rbp-30h]

  v3 = *(_QWORD *)(a1 + 8);
  LODWORD(v16) = a2;
  v4 = *(_QWORD *)(v3 + 992);
  v5 = v3 + 984;
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
    if ( v3 + 984 != v5 && a2 >= *(_DWORD *)(v5 + 32) )
    {
      result = *(_QWORD *)(v5 + 40);
      if ( result )
        return result;
LABEL_11:
      v9 = *(_QWORD *)(a1 + 8);
      v10 = *(_QWORD *)(v9 + 16);
      v11 = *(_QWORD *)(v10 + 88);
      v12 = *(_QWORD *)(v10 + 80);
      LODWORD(v16) = a2;
      v13[0] = v12;
      v15 = 773;
      v14[0] = v13;
      v14[1] = "line_table_start";
      v17[0] = (unsigned int *)v14;
      v13[1] = v11;
      v17[1] = v16;
      v18 = 2306;
      result = sub_38BF510(v9, (__int64)v17);
      *(_QWORD *)(v5 + 40) = result;
      return result;
    }
  }
  v17[0] = (unsigned int *)&v16;
  v5 = sub_38E1100((_QWORD *)(v3 + 976), v5, v17);
  result = *(_QWORD *)(v5 + 40);
  if ( !result )
    goto LABEL_11;
  return result;
}
