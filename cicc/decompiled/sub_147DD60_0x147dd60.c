// Function: sub_147DD60
// Address: 0x147dd60
//
unsigned __int64 __fastcall sub_147DD60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int16 v7; // ax
  __int64 v8; // r12
  unsigned int v9; // r13d
  int v10; // eax
  unsigned __int64 result; // rax
  __int64 v12; // rcx
  unsigned int v13; // ecx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // r12d

  v3 = sub_1474160(a1, a2, a3);
  if ( v3 == sub_1456E90(a1) )
    return 1;
  v4 = sub_1456040(v3);
  v5 = sub_145CF80(a1, v4, 1, 0);
  v6 = sub_13A5B00(a1, v3, v5, 0, 0);
  v7 = *(_WORD *)(v6 + 24);
  if ( v7 == 4 )
  {
    if ( *(_QWORD *)(v6 + 40) != 2 )
    {
LABEL_16:
      v13 = sub_14687F0(a1, v6);
      result = (unsigned int)(1 << v13);
      if ( v13 >= 0x1F )
        return 0x80000000LL;
      return result;
    }
    v14 = sub_1456FE0(**(_QWORD **)(v6 + 32));
    v15 = sub_1456FE0(*(_QWORD *)(*(_QWORD *)(v6 + 32) + 8LL));
    if ( v14 && v14 == v15 )
    {
      if ( *(_WORD *)(v14 + 24) )
        return 1;
      v16 = *(_QWORD *)(v14 + 32);
      if ( !v16 )
        return 1;
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 - (unsigned int)sub_1455840(v16 + 24) > 0x20 )
        return 1;
      result = *(_QWORD *)(v16 + 24);
      if ( v17 > 0x40 )
        return *(_QWORD *)result;
      return result;
    }
    v7 = *(_WORD *)(v6 + 24);
  }
  if ( v7 )
    goto LABEL_16;
  v8 = *(_QWORD *)(v6 + 32);
  if ( !v8 )
    return 1;
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 > 0x40 )
  {
    v10 = sub_16A57B0(v8 + 24);
    if ( v9 - v10 <= 0x20 && v9 != v10 )
    {
      result = *(_QWORD *)(v8 + 24);
      return *(_QWORD *)result;
    }
    return 1;
  }
  result = *(_QWORD *)(v8 + 24);
  if ( !result )
    return 1;
  _BitScanReverse64((unsigned __int64 *)&v12, result);
  if ( 64 - ((unsigned int)v12 ^ 0x3F) > 0x20 || v9 == ((unsigned int)v12 ^ 0x3F) + v9 - 64 )
    return 1;
  return result;
}
