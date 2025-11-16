// Function: sub_1AB3AB0
// Address: 0x1ab3ab0
//
bool __fastcall sub_1AB3AB0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi
  bool result; // al
  int v9; // r13d
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // rdi
  _QWORD v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v17[0] = a1;
  v5 = *(_QWORD *)(a1 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = *(_QWORD *)(a2 + 24);
  v7 = **(_QWORD **)(v6 + 16);
  if ( v5 != v7 )
  {
    result = sub_15FBC60(v7, v5);
    if ( !result )
    {
      if ( a3 )
      {
        *a3 = "Return type mismatch";
        return result;
      }
      return 0;
    }
    v6 = *(_QWORD *)(a2 + 24);
  }
  v9 = *(_DWORD *)(v6 + 12);
  v10 = sub_1389B50(v17);
  v11 = v17[0];
  if ( -1431655765
     * (unsigned int)((__int64)(v10
                              - ((v17[0] & 0xFFFFFFFFFFFFFFF8LL)
                               - 24LL * (*(_DWORD *)((v17[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3) != v9 - 1
    && !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
  {
    if ( a3 )
      *a3 = "The number of arguments mismatch";
    return 0;
  }
  if ( v9 == 1 )
    return 1;
  v12 = (unsigned int)(v9 - 2);
  v13 = 0;
  v14 = 8 * v12;
  while ( 1 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 16LL) + v13 + 8);
    v16 = **(_QWORD **)((v11 & 0xFFFFFFFFFFFFFFF8LL)
                      + 3 * v13
                      - 24LL * (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
    if ( v16 != v15 )
    {
      result = sub_15FBC60(v16, v15);
      if ( !result )
        break;
    }
    if ( v14 == v13 )
      return 1;
    v11 = v17[0];
    v13 += 8;
  }
  if ( !a3 )
    return 0;
  *a3 = "Argument type mismatch";
  return result;
}
