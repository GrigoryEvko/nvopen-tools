// Function: sub_1378260
// Address: 0x1378260
//
__int64 __fastcall sub_1378260(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // r14
  int v6; // r13d
  unsigned int v7; // ebx
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rsi
  unsigned int v18; // edi
  _QWORD *v19; // rcx
  __int64 v20; // rax
  int v21; // [rsp+4h] [rbp-5Ch]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v25[7]; // [rsp+28h] [rbp-38h] BYREF

  v22 = sub_157EBA0(a2);
  result = sub_15F4D60(v22);
  if ( !(_DWORD)result )
    return result;
  v3 = a1 + 240;
  v4 = sub_157EBA0(a2);
  if ( !v4 )
    goto LABEL_30;
  v21 = sub_15F4D60(v4);
  v5 = sub_157EBA0(a2);
  v6 = v21 >> 2;
  v3 = a1 + 240;
  if ( v21 >> 2 > 0 )
  {
    v7 = 0;
    while ( 1 )
    {
      v12 = sub_15F4DF0(v5, v7);
      if ( !sub_1377F70(v3, v12) )
        goto LABEL_11;
      v8 = v7 + 1;
      v9 = sub_15F4DF0(v5, v7 + 1);
      if ( !sub_1377F70(v3, v9)
        || (v8 = v7 + 2, v10 = sub_15F4DF0(v5, v7 + 2), !sub_1377F70(v3, v10))
        || (v8 = v7 + 3, v11 = sub_15F4DF0(v5, v7 + 3), !sub_1377F70(v3, v11)) )
      {
        v7 = v8;
        goto LABEL_11;
      }
      v7 += 4;
      if ( !--v6 )
      {
        v14 = v21 - v7;
        goto LABEL_26;
      }
    }
  }
  v14 = v21;
  v7 = 0;
LABEL_26:
  if ( v14 == 2 )
    goto LABEL_27;
  if ( v14 == 3 )
  {
    v20 = sub_15F4DF0(v5, v7);
    if ( !sub_1377F70(v3, v20) )
      goto LABEL_11;
    ++v7;
LABEL_27:
    v15 = sub_15F4DF0(v5, v7);
    if ( !sub_1377F70(v3, v15) )
      goto LABEL_11;
    ++v7;
    goto LABEL_29;
  }
  if ( v14 != 1 )
    goto LABEL_30;
LABEL_29:
  v16 = sub_15F4DF0(v5, v7);
  if ( sub_1377F70(v3, v16) )
    goto LABEL_30;
LABEL_11:
  if ( v7 == v21 )
  {
LABEL_30:
    result = *(_QWORD *)(a1 + 248);
    if ( *(_QWORD *)(a1 + 256) != result )
      return sub_16CCBA0(v3, a2);
    v17 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 268));
    v18 = *(_DWORD *)(a1 + 268);
    if ( (__int64 *)result == v17 )
      goto LABEL_54;
    v19 = 0;
    while ( a2 != *(_QWORD *)result )
    {
      if ( *(_QWORD *)result == -2 )
        v19 = (_QWORD *)result;
      result += 8;
      if ( v17 == (__int64 *)result )
      {
LABEL_37:
        if ( !v19 )
          goto LABEL_54;
        *v19 = a2;
        --*(_DWORD *)(a1 + 272);
        ++*(_QWORD *)(a1 + 240);
        return a1;
      }
    }
    return result;
  }
  if ( *(_BYTE *)(v22 + 16) != 29 || !sub_1377F70(v3, *(_QWORD *)(v22 - 48)) )
  {
    result = a2;
    v13 = *(_QWORD *)(a2 + 48);
    if ( v13 == a2 + 40 )
      return result;
    while ( 1 )
    {
      if ( !v13 )
        BUG();
      if ( *(_BYTE *)(v13 - 8) == 78 )
      {
        if ( (unsigned __int8)sub_1560260(v13 + 32, 0xFFFFFFFFLL, 7) )
          break;
        result = *(_QWORD *)(v13 - 48);
        if ( !*(_BYTE *)(result + 16) )
        {
          v25[0] = *(_QWORD *)(result + 112);
          result = sub_1560260(v25, 0xFFFFFFFFLL, 7);
          if ( (_BYTE)result )
            break;
        }
      }
      v13 = *(_QWORD *)(v13 + 8);
      if ( a2 + 40 == v13 )
        return result;
    }
    result = *(_QWORD *)(a1 + 248);
    if ( *(_QWORD *)(a1 + 256) != result )
      return sub_16CCBA0(v3, a2);
    v17 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 268));
    v18 = *(_DWORD *)(a1 + 268);
    if ( (__int64 *)result != v17 )
    {
      v19 = 0;
      while ( a2 != *(_QWORD *)result )
      {
        if ( *(_QWORD *)result == -2 )
          v19 = (_QWORD *)result;
        result += 8;
        if ( v17 == (__int64 *)result )
          goto LABEL_37;
      }
      return result;
    }
LABEL_54:
    result = a1;
    if ( v18 < *(_DWORD *)(a1 + 264) )
    {
      *(_DWORD *)(a1 + 268) = v18 + 1;
      *v17 = a2;
      ++*(_QWORD *)(a1 + 240);
      return result;
    }
    return sub_16CCBA0(v3, a2);
  }
  result = *(_QWORD *)(a1 + 248);
  if ( *(_QWORD *)(a1 + 256) != result )
    return sub_16CCBA0(v3, a2);
  v17 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 268));
  v18 = *(_DWORD *)(a1 + 268);
  if ( (__int64 *)result == v17 )
    goto LABEL_54;
  v19 = 0;
  while ( a2 != *(_QWORD *)result )
  {
    if ( *(_QWORD *)result == -2 )
      v19 = (_QWORD *)result;
    result += 8;
    if ( v17 == (__int64 *)result )
      goto LABEL_37;
  }
  return result;
}
