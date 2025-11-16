// Function: sub_38F38E0
// Address: 0x38f38e0
//
char __fastcall sub_38F38E0(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdx
  unsigned int v6; // ecx
  char result; // al
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-58h] BYREF
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+18h] [rbp-48h]
  _QWORD v20[2]; // [rsp+20h] [rbp-40h] BYREF
  char v21; // [rsp+30h] [rbp-30h]
  char v22; // [rsp+31h] [rbp-2Fh]

  v2 = *a1;
  v18 = 0;
  v19 = 0;
  v3 = sub_3909460(v2);
  v4 = sub_39092A0(v3);
  result = sub_38F0EE0(*a1, &v18, v5, v6);
  if ( result )
  {
    v11 = *a1;
    v22 = 1;
    v20[0] = "unexpected token in '.cv_loc' directive";
    v21 = 3;
    return sub_3909CF0(v11, v20, 0, 0, v8, v9);
  }
  if ( v19 == 12 )
  {
    if ( *(_QWORD *)v18 == 0x6575676F6C6F7270LL && *(_DWORD *)(v18 + 8) == 1684956511 )
    {
      *(_BYTE *)a1[1] = 1;
      return result;
    }
    goto LABEL_4;
  }
  if ( v19 != 7 || *(_DWORD *)v18 != 1935635305 || *(_WORD *)(v18 + 4) != 28020 || *(_BYTE *)(v18 + 6) != 116 )
  {
LABEL_4:
    v10 = *a1;
    v22 = 1;
    v20[0] = "unknown sub-directive in '.cv_loc' directive";
    v21 = 3;
    return sub_3909790(v10, v4, v20, 0, 0);
  }
  v12 = sub_3909460(*a1);
  v13 = sub_39092A0(v12);
  v14 = *a1;
  v20[0] = 0;
  v15 = v13;
  result = sub_38EB6A0(v14, &v17, (__int64)v20);
  if ( !result )
  {
    *(_QWORD *)a1[2] = -1;
    if ( *(_DWORD *)v17 == 1 )
      *(_QWORD *)a1[2] = *(_QWORD *)(v17 + 16);
    if ( *(_QWORD *)a1[2] > 1u )
    {
      v16 = *a1;
      v22 = 1;
      v20[0] = "is_stmt value not 0 or 1";
      v21 = 3;
      return sub_3909790(v16, v15, v20, 0, 0);
    }
  }
  return result;
}
