// Function: sub_29A3A40
// Address: 0x29a3a40
//
__int64 __fastcall sub_29A3A40(unsigned __int8 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 result; // rax
  int v11; // edx
  unsigned int v12; // r12d
  unsigned int v13; // r14d
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r15
  char v21; // r12
  char v22; // r12
  __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+20h] [rbp-60h]
  __int64 v30; // [rsp+28h] [rbp-58h]
  __int64 v31; // [rsp+38h] [rbp-48h]
  int v32; // [rsp+38h] [rbp-48h]
  _QWORD v33[7]; // [rsp+48h] [rbp-38h] BYREF

  v5 = sub_B2BEC0(a2);
  v6 = *((_QWORD *)a1 + 1);
  v29 = v5;
  v7 = v5;
  v8 = *(_QWORD *)(a2 + 24);
  v9 = **(_QWORD **)(v8 + 16);
  if ( v6 != v9 )
  {
    result = sub_B50C50(v9, v6, v7);
    if ( !(_BYTE)result )
    {
      if ( a3 )
      {
        *a3 = "Return type mismatch";
        return result;
      }
      return 0;
    }
    v8 = *(_QWORD *)(a2 + 24);
  }
  v11 = *a1;
  v12 = *(_DWORD *)(v8 + 12) - 1;
  v13 = v12;
  if ( v11 == 40 )
  {
    v14 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_15;
  }
  else
  {
    v14 = 0;
    if ( v11 != 85 )
    {
      v14 = 64;
      if ( v11 != 34 )
        BUG();
    }
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_15;
  }
  v15 = sub_BD2BC0((__int64)a1);
  v31 = v16 + v15;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v31 >> 4) )
LABEL_53:
      BUG();
LABEL_15:
    v19 = 0;
    goto LABEL_16;
  }
  if ( !(unsigned int)((v31 - sub_BD2BC0((__int64)a1)) >> 4) )
    goto LABEL_15;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_53;
  v32 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v17 = sub_BD2BC0((__int64)a1);
  v19 = 32LL * (unsigned int)(*(_DWORD *)(v17 + v18 - 4) - v32);
LABEL_16:
  v27 = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v14 - v19) >> 5;
  if ( v12 != (_DWORD)v27 && !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
  {
    if ( a3 )
    {
      *a3 = "The number of arguments mismatch";
      return 0;
    }
    return 0;
  }
  if ( !v12 )
  {
LABEL_38:
    if ( v13 >= (unsigned int)v27 )
      return 1;
    while ( !(unsigned __int8)sub_B49B80((__int64)a1, v13, 85) )
    {
      if ( ++v13 == (_DWORD)v27 )
        return 1;
    }
    if ( a3 )
    {
      *a3 = "SRet arg to vararg function";
      return 0;
    }
    return 0;
  }
  v20 = 0;
  v30 = v12;
  while ( 1 )
  {
    v13 = v20 + 1;
    v21 = sub_B2D640(a2, v20, 81);
    v33[0] = *((_QWORD *)a1 + 9);
    if ( v21 != (unsigned __int8)sub_A74710(v33, (int)v20 + 1, 81) )
    {
      if ( !a3 )
        return 0;
      *a3 = "byval mismatch";
      return 0;
    }
    v22 = sub_B2D640(a2, v20, 83);
    v33[0] = *((_QWORD *)a1 + 9);
    if ( v22 != (unsigned __int8)sub_A74710(v33, v13, 83) )
    {
      if ( !a3 )
        return 0;
      *a3 = "inalloca mismatch";
      return 0;
    }
    v23 = v20 + 1;
    v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 16LL) + 8 * (v20 + 1));
    v25 = *(_QWORD *)&a1[32 * (v20 - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
    v26 = *(_QWORD *)(v25 + 8);
    if ( v26 != v24 )
      break;
LABEL_22:
    v20 = v23;
    if ( v30 == v23 )
      goto LABEL_38;
  }
  result = sub_B50C50(*(_QWORD *)(v25 + 8), v24, v29);
  if ( (_BYTE)result )
  {
    if ( sub_B49200((__int64)a1)
      && (*(_BYTE *)(v24 + 8) != 14
       || *(_BYTE *)(v26 + 8) != 14
       || *(_DWORD *)(v26 + 8) >> 8 != *(_DWORD *)(v24 + 8) >> 8) )
    {
      if ( a3 )
        *a3 = "Musttail call Argument type mismatch";
      return 0;
    }
    goto LABEL_22;
  }
  if ( !a3 )
    return 0;
  *a3 = "Argument type mismatch";
  return result;
}
