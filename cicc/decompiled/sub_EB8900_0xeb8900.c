// Function: sub_EB8900
// Address: 0xeb8900
//
char __fastcall sub_EB8900(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  char result; // al
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  __int64 v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+18h] [rbp-58h]
  _QWORD v16[4]; // [rsp+20h] [rbp-50h] BYREF
  char v17; // [rsp+40h] [rbp-30h]
  char v18; // [rsp+41h] [rbp-2Fh]

  v2 = *(_QWORD *)a1;
  v14 = 0;
  v15 = 0;
  v3 = sub_ECD7B0(v2);
  v4 = sub_ECD6A0(v3);
  result = sub_EB61F0(*(_QWORD *)a1, &v14);
  if ( result )
  {
    v7 = *(_QWORD *)a1;
    v18 = 1;
    v16[0] = "unexpected token in '.cv_loc' directive";
    v17 = 3;
    return sub_ECE0E0(v7, v16, 0, 0);
  }
  if ( v15 == 12 )
  {
    if ( *(_QWORD *)v14 == 0x6575676F6C6F7270LL && *(_DWORD *)(v14 + 8) == 1684956511 )
    {
      **(_BYTE **)(a1 + 8) = 1;
      return result;
    }
    v6 = *(_QWORD *)a1;
LABEL_4:
    v18 = 1;
    v16[0] = "unknown sub-directive in '.cv_loc' directive";
    v17 = 3;
    return sub_ECDA70(v6, v4, v16, 0, 0);
  }
  v6 = *(_QWORD *)a1;
  if ( v15 != 7 || *(_DWORD *)v14 != 1935635305 || *(_WORD *)(v14 + 4) != 28020 || *(_BYTE *)(v14 + 6) != 116 )
    goto LABEL_4;
  v8 = sub_ECD7B0(v6);
  v9 = sub_ECD6A0(v8);
  v10 = *(_QWORD *)a1;
  v16[0] = 0;
  v11 = v9;
  result = sub_EAC4D0(v10, &v13, (__int64)v16);
  if ( !result )
  {
    **(_QWORD **)(a1 + 16) = -1;
    if ( *(_BYTE *)v13 == 1 )
      **(_QWORD **)(a1 + 16) = *(_QWORD *)(v13 + 16);
    if ( **(_QWORD **)(a1 + 16) > 1u )
    {
      v12 = *(_QWORD *)a1;
      v18 = 1;
      v16[0] = "is_stmt value not 0 or 1";
      v17 = 3;
      return sub_ECDA70(v12, v11, v16, 0, 0);
    }
  }
  return result;
}
