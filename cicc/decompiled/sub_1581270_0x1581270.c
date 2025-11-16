// Function: sub_1581270
// Address: 0x1581270
//
__int64 __fastcall sub_1581270(__int64 a1, __int64 a2)
{
  char v2; // al
  char v4; // dl
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // [rsp-20h] [rbp-20h]

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 1 )
    return 42;
  if ( *(_BYTE *)(a2 + 16) == 1 )
    return 42;
  v4 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( v4 == 9 || v4 == 4 )
    return 42;
  if ( v2 == 3 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    v6 = *(unsigned __int8 *)(v5 + 8);
    if ( (unsigned __int8)v6 > 0xFu || (v7 = 35454, !_bittest64(&v7, v6)) )
    {
      if ( (unsigned int)(v6 - 13) > 1 && (_DWORD)v6 != 16 || !(unsigned __int8)sub_16435F0(*(_QWORD *)(a1 + 24), 0) )
        return 42;
    }
    if ( (unsigned __int8)sub_1642FB0(v5) )
      return 42;
  }
  v8 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( v8 == 9 || v8 == 4 )
    return 42;
  if ( *(_BYTE *)(a2 + 16) == 3 )
  {
    v9 = *(_QWORD *)(a2 + 24);
    v10 = *(unsigned __int8 *)(v9 + 8);
    if ( (unsigned __int8)v10 > 0xFu || (v12 = 35454, !_bittest64(&v12, v10)) )
    {
      if ( (unsigned int)(v10 - 13) > 1 && (_DWORD)v10 != 16 )
        return 42;
      v13 = *(_QWORD *)(a2 + 24);
      v11 = sub_16435F0(v9, 0);
      v9 = v13;
      if ( !v11 )
        return 42;
    }
    if ( (unsigned __int8)sub_1642FB0(v9) )
      return 42;
  }
  return 33;
}
