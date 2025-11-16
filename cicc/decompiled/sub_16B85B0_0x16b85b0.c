// Function: sub_16B85B0
// Address: 0x16b85b0
//
__int64 __fastcall sub_16B85B0(const char **a1, __int64 a2, __int64 a3)
{
  const char **v3; // r14
  __int64 v5; // r12
  size_t v6; // rdx
  char v7; // bl
  __int64 result; // rax
  const char *v9; // rdx
  const char *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  const char *v16; // rbx
  const char *v17; // r15
  __int64 v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+8h] [rbp-88h]
  const char *v20; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  char v22; // [rsp+20h] [rbp-70h] BYREF
  const char *v23; // [rsp+30h] [rbp-60h] BYREF
  size_t v24; // [rsp+38h] [rbp-58h]
  _QWORD v25[2]; // [rsp+40h] [rbp-50h] BYREF
  int v26; // [rsp+50h] [rbp-40h]
  const char **v27; // [rsp+58h] [rbp-38h]

  v3 = a1;
  v5 = a2;
  v6 = *(_QWORD *)(a2 + 32);
  if ( !v6
    || (a2 = *(_QWORD *)(a2 + 24),
        v24 = v6,
        a1 = (const char **)(a3 + 128),
        v25[0] = v5,
        v23 = (const char *)a2,
        sub_16B8110(a3 + 128, (const void *)a2, v6, v25),
        (_BYTE)v6) )
  {
    v7 = 0;
  }
  else
  {
    v9 = v3[1];
    v10 = *v3;
    v20 = &v22;
    v27 = &v20;
    v21 = 0;
    v22 = 0;
    v26 = 1;
    v25[1] = 0;
    v25[0] = 0;
    v24 = 0;
    v23 = (const char *)&unk_49EFBE0;
    v11 = sub_16E7EE0(&v23, v10, v9);
    v12 = sub_1263B40(v11, ": CommandLine Error: Option '");
    v13 = sub_1549FF0(v12, *(const char **)(v5 + 24), *(_QWORD *)(v5 + 32));
    sub_1263B40(v13, "' registered more than once!\n");
    sub_16E7BC0(&v23);
    a2 = 1;
    sub_1C3EFD0(&v20, 1);
    a1 = &v20;
    v7 = 1;
    sub_2240A30(&v20);
  }
  if ( ((*(_WORD *)(v5 + 12) >> 7) & 3) == 1 )
  {
    result = *(unsigned int *)(a3 + 40);
    if ( (unsigned int)result >= *(_DWORD *)(a3 + 44) )
    {
      sub_16CD150(a3 + 32, a3 + 48, 0, 8);
      result = *(unsigned int *)(a3 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * result) = v5;
    ++*(_DWORD *)(a3 + 40);
    if ( v7 )
      return result;
LABEL_7:
    result = sub_16B4B80((__int64)&unk_4FA0170);
    if ( a3 == result )
    {
      sub_16B5AB0(&v20, (__int64 *)v3 + 30, v3[32]);
      v16 = v20;
      v18 = v21;
      sub_16B55A0(&v23, (__int64 *)v3 + 30);
      v17 = v23;
      result = v18;
      while ( v17 != v16 )
      {
        if ( a3 != *(_QWORD *)v16 )
        {
          v19 = result;
          sub_16B85B0(v3, v5);
          result = v19;
        }
        for ( v16 += 8; (const char *)result != v16; v16 += 8 )
        {
          if ( *(_QWORD *)v16 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
      }
    }
    return result;
  }
  if ( (*(_BYTE *)(v5 + 13) & 8) != 0 )
  {
    result = *(unsigned int *)(a3 + 88);
    if ( (unsigned int)result >= *(_DWORD *)(a3 + 92) )
    {
      sub_16CD150(a3 + 80, a3 + 96, 0, 8);
      result = *(unsigned int *)(a3 + 88);
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 80) + 8 * result) = v5;
    ++*(_DWORD *)(a3 + 88);
    if ( v7 )
      return result;
    goto LABEL_7;
  }
  result = *(_BYTE *)(v5 + 12) & 7;
  if ( (_BYTE)result == 4 )
  {
    if ( *(_QWORD *)(a3 + 160) )
    {
      v14 = sub_16E8CB0(a1, a2, v6);
      v23 = "Cannot specify more than one option with cl::ConsumeAfter!";
      LOWORD(v25[0]) = 259;
      result = sub_16B1F90(v5, (__int64)&v23, 0, 0, v14, v15);
      *(_QWORD *)(a3 + 160) = v5;
      return result;
    }
    *(_QWORD *)(a3 + 160) = v5;
  }
  if ( !v7 )
    goto LABEL_7;
  return result;
}
