// Function: sub_18498D0
// Address: 0x18498d0
//
__int64 __fastcall sub_18498D0(__int64 *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rbx
  __int64 v6; // r12
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // eax
  __int64 v15; // rdi
  int v16; // r8d
  int v17; // ecx
  _QWORD v18[4]; // [rsp-20h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 <= 0x17u )
    return 0;
  if ( v2 == 78 )
  {
    v4 = a2 | 4;
  }
  else
  {
    if ( v2 != 29 )
      return 0;
    v4 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v6 = *a1;
  v7 = (_QWORD *)(v5 + 56);
  if ( (v4 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v7, -1, 8) )
    {
      v8 = *(_QWORD *)(v5 - 24);
      if ( *(_BYTE *)(v8 + 16) )
        return 0;
      v18[0] = *(_QWORD *)(v8 + 112);
      if ( !(unsigned __int8)sub_1560260(v18, -1, 8) )
        return 0;
    }
    v10 = v5 - 24;
  }
  else
  {
    if ( !(unsigned __int8)sub_1560260(v7, -1, 8) )
    {
      v9 = *(_QWORD *)(v5 - 72);
      if ( *(_BYTE *)(v9 + 16) )
        return 0;
      v18[0] = *(_QWORD *)(v9 + 112);
      if ( !(unsigned __int8)sub_1560260(v18, -1, 8) )
        return 0;
    }
    v10 = v5 - 72;
  }
  v11 = *(_QWORD *)v10;
  if ( *(_BYTE *)(*(_QWORD *)v10 + 16LL) )
    v11 = 0;
  if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
  {
    v12 = v6 + 16;
    v13 = 7;
  }
  else
  {
    v17 = *(_DWORD *)(v6 + 24);
    v12 = *(_QWORD *)(v6 + 16);
    result = 1;
    if ( !v17 )
      return result;
    v13 = v17 - 1;
  }
  v14 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v15 = *(_QWORD *)(v12 + 8LL * v14);
  if ( v15 == v11 )
    return 0;
  v16 = 1;
  while ( v15 != -8 )
  {
    v14 = v13 & (v16 + v14);
    v15 = *(_QWORD *)(v12 + 8LL * v14);
    if ( v11 == v15 )
      return 0;
    ++v16;
  }
  return 1;
}
