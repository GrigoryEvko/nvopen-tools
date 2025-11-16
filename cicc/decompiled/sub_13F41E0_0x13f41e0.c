// Function: sub_13F41E0
// Address: 0x13f41e0
//
unsigned __int64 __fastcall sub_13F41E0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  int v5; // eax
  unsigned __int8 *v6; // r14
  unsigned __int64 result; // rax
  __int64 v8; // r15
  unsigned int v9; // r13d
  int v10; // eax
  unsigned int v11; // edx
  bool v12; // cl
  __int64 v13; // rax
  int v14; // r9d
  unsigned int v15; // r13d
  bool v16; // r13
  _BYTE *v17; // rax
  __int64 v18; // rdi
  int v19; // [rsp+Ch] [rbp-64h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  unsigned int v21; // [rsp+10h] [rbp-60h]
  int v22; // [rsp+18h] [rbp-58h]
  const char *v23; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]
  unsigned int v26; // [rsp+38h] [rbp-38h]

  v3 = a1[23];
  v20 = a1[24];
  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  v6 = *(unsigned __int8 **)(a2 - 24);
  v22 = v5;
  result = v6[16];
  if ( (_BYTE)result != 9 )
  {
    v8 = *(_QWORD *)v6;
    if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 16 )
    {
      if ( (unsigned __int8)result > 0x10u )
        return result;
      if ( !(unsigned __int8)sub_1595F50(v6) )
      {
        result = *(_QWORD *)(v8 + 32);
        v19 = result;
        if ( !(_DWORD)result )
          return result;
        v9 = 0;
        while ( 1 )
        {
          v13 = sub_15A0A60(v6, v9);
          if ( *(_BYTE *)(v13 + 16) == 9 )
            break;
          sub_14C2530((unsigned int)&v23, v13, v22, 0, 0, 0, 0, 0);
          v11 = v24;
          if ( v24 > 0x40 )
          {
            v21 = v24;
            v10 = sub_16A58F0(&v23);
            v11 = v21;
            v12 = v21 == v10;
          }
          else
          {
            v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == (_QWORD)v23;
          }
          result = v26;
          if ( v12 )
          {
            if ( v26 > 0x40 && v25 )
            {
              j_j___libc_free_0_0(v25);
              v11 = v24;
            }
            if ( v11 > 0x40 && v23 )
              j_j___libc_free_0_0(v23);
            goto LABEL_27;
          }
          if ( v26 > 0x40 && v25 )
          {
            result = j_j___libc_free_0_0(v25);
            v11 = v24;
          }
          if ( v11 > 0x40 )
          {
            if ( v23 )
              result = j_j___libc_free_0_0(v23);
          }
          if ( v19 == ++v9 )
            return result;
        }
      }
      goto LABEL_27;
    }
    v14 = 0;
    if ( (unsigned __int8)result > 0x17u )
      v14 = *(_QWORD *)(a2 - 24);
    sub_14C2530((unsigned int)&v23, (_DWORD)v6, v22, 0, v3, v14, v20, 0);
    v15 = v24;
    if ( v24 <= 0x40 )
    {
      result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24);
      v16 = result == (_QWORD)v23;
      if ( v26 <= 0x40 )
        goto LABEL_25;
      v18 = v25;
      if ( !v25 )
        goto LABEL_25;
    }
    else
    {
      result = sub_16A58F0(&v23);
      v16 = v15 == (_DWORD)result;
      if ( v26 <= 0x40 || (v18 = v25) == 0 )
      {
LABEL_23:
        if ( v23 )
          result = j_j___libc_free_0_0(v23);
LABEL_25:
        if ( !v16 )
          return result;
        goto LABEL_27;
      }
    }
    result = j_j___libc_free_0_0(v18);
    if ( v24 <= 0x40 )
      goto LABEL_25;
    goto LABEL_23;
  }
LABEL_27:
  v23 = "Undefined behavior: Division by zero";
  LOWORD(v25) = 259;
  sub_16E2CE0(&v23, a1 + 30);
  v17 = (_BYTE *)a1[33];
  if ( (unsigned __int64)v17 >= a1[32] )
  {
    sub_16E7DE0(a1 + 30, 10);
  }
  else
  {
    a1[33] = v17 + 1;
    *v17 = 10;
  }
  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
  {
    sub_15537D0(a2, a1 + 30, 1);
    result = a1[33];
    if ( result < a1[32] )
      goto LABEL_31;
  }
  else
  {
    sub_155C2B0(a2, a1 + 30, 0);
    result = a1[33];
    if ( result < a1[32] )
    {
LABEL_31:
      a1[33] = result + 1;
      *(_BYTE *)result = 10;
      return result;
    }
  }
  return sub_16E7DE0(a1 + 30, 10);
}
