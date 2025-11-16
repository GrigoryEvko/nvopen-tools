// Function: sub_EF7190
// Address: 0xef7190
//
unsigned __int64 __fastcall sub_EF7190(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 result; // rax
  __int64 v8; // r9
  __int64 *v9; // r12
  __int64 v10; // rdx
  unsigned __int8 *v11; // r13
  unsigned __int8 *v12; // r8
  __int64 **v13; // rsi
  __int64 v14; // r12
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  __int64 v22; // rcx
  char v23; // dl
  char v24; // [rsp+8h] [rbp-E8h]
  __int64 *v25; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 *v27; // [rsp+28h] [rbp-C8h] BYREF
  __int64 *v28; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+38h] [rbp-B8h]
  _BYTE v30[176]; // [rsp+40h] [rbp-B0h] BYREF

  *(_QWORD *)(a1 + 8) = a2 + a3;
  v4 = *(_QWORD *)(a1 + 16);
  *(_BYTE *)(a1 + 937) = a4;
  *(_QWORD *)(a1 + 24) = v4;
  v5 = *(_QWORD *)(a1 + 296);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 304) = v5;
  v6 = *(_QWORD *)(a1 + 664);
  *(_WORD *)(a1 + 776) = 1;
  *(_QWORD *)(a1 + 672) = v6;
  *(_QWORD *)(a1 + 784) = -1;
  *(_QWORD *)(a1 + 792) = 0;
  *(_DWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 920) = 0;
  if ( a3 <= 1
    || *(_WORD *)a2 != 23135
    && (a3 == 2
     || (*(_WORD *)a2 != 24415 || *(_BYTE *)(a2 + 2) != 90)
     && (a3 == 3 || *(_DWORD *)a2 != 1516199775 && (a3 == 4 || *(_DWORD *)a2 != 1600085855 || *(_BYTE *)(a2 + 4) != 90))) )
  {
    v29 = a2;
    v28 = (__int64 *)a3;
    return sub_EE6A90(a1 + 808, (__int64 *)&v28);
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, &unk_3C1BC40)
    || (unsigned __int8)sub_EE3B50((const void **)a1, 3u, "__Z") )
  {
    result = (unsigned __int64)sub_EF05F0((unsigned __int8 **)a1, 1);
    v9 = (__int64 *)result;
    if ( !result )
      return 0;
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *(unsigned __int8 **)a1;
    if ( v10 != *(_QWORD *)a1 )
    {
      if ( *v11 != 46 )
        return 0;
      v12 = *(unsigned __int8 **)a1;
      v25 = (__int64 *)(v10 - (_QWORD)v11);
      v24 = *(_BYTE *)(a1 + 937);
      v29 = 0x2000000000LL;
      v28 = (__int64 *)v30;
      sub_EE3E30((__int64)&v28, 1u, result, v10 - (_QWORD)v11, (__int64)v12, v8);
      v13 = &v28;
      result = (unsigned __int64)sub_C65B40(a1 + 904, (__int64)&v28, (__int64 *)&v27, (__int64)off_497B2F0);
      if ( result )
      {
        v14 = result + 8;
        if ( v28 != (__int64 *)v30 )
          _libc_free(v28, &v28);
        v28 = (__int64 *)v14;
        v15 = sub_EE6840(a1 + 944, (__int64 *)&v28);
        if ( v15 )
        {
          v16 = v15[1];
          if ( v16 )
            v14 = v16;
        }
        result = v14;
        if ( *(_QWORD *)(a1 + 928) == v14 )
          *(_BYTE *)(a1 + 936) = 1;
      }
      else
      {
        if ( v24 )
        {
          v13 = (__int64 **)sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
          *v13 = 0;
          v23 = *((_BYTE *)v13 + 18);
          *((_WORD *)v13 + 8) = 16385;
          v13[3] = v9;
          v13[4] = v25;
          *((_BYTE *)v13 + 18) = v23 & 0xF0 | 5;
          v13[5] = (__int64 *)v11;
          v13[1] = (__int64 *)&unk_49DEDC8;
          sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v13, v27, (__int64)off_497B2F0);
          result = (unsigned __int64)(v13 + 1);
        }
        if ( v28 != (__int64 *)v30 )
        {
          v26 = result;
          _libc_free(v28, v13);
          result = v26;
        }
        *(_QWORD *)(a1 + 920) = result;
      }
      *(_QWORD *)a1 = *(_QWORD *)(a1 + 8);
    }
  }
  else
  {
    if ( (unsigned __int8)sub_EE3B50((const void **)a1, 4u, "___Z")
      || (unsigned __int8)sub_EE3B50((const void **)a1, 5u, "____Z") )
    {
      v28 = sub_EF05F0((unsigned __int8 **)a1, 1);
      if ( !v28 || !(unsigned __int8)sub_EE3B50((const void **)a1, 0xDu, "_block_invoke") )
        return 0;
      v21 = *(unsigned __int8 **)a1;
      if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v21 != 95 )
      {
        sub_EE32C0((char **)a1, 0);
      }
      else
      {
        *(_QWORD *)a1 = v21 + 1;
        result = sub_EE32C0((char **)a1, 0);
        if ( !result )
          return result;
      }
      v22 = *(_QWORD *)(a1 + 8);
      if ( v22 != *(_QWORD *)a1 )
      {
        result = 0;
        if ( **(_BYTE **)a1 != 46 )
          return result;
        *(_QWORD *)a1 = v22;
      }
      return sub_EE8500(a1 + 808, "invocation function for block in ", (__int64 *)&v28);
    }
    result = sub_EF1F20(a1, 5, v17, v18, v19, v20);
    if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
      return 0;
  }
  return result;
}
