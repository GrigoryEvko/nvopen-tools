// Function: sub_140B650
// Address: 0x140b650
//
__int64 __fastcall sub_140B650(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rsi
  int v7; // esi
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  char v15; // [rsp+Bh] [rbp-25h] BYREF
  unsigned int v16[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = sub_140ABA0(a1, 0, &v15);
  if ( !v3 )
    return 0;
  if ( v15 )
    return 0;
  v4 = v3;
  v6 = sub_1649960(v3);
  if ( !a2
    || !(unsigned __int8)sub_149B630(*a2, v6, v5, v16)
    || (((int)*(unsigned __int8 *)(*a2 + (signed int)v16[0] / 4) >> (2 * (v16[0] & 3))) & 3) == 0 )
  {
    return 0;
  }
  if ( v16[0] <= 0x1C )
  {
    v13 = 272777360;
    if ( _bittest64(&v13, v16[0]) )
    {
      v7 = 1;
      goto LABEL_8;
    }
  }
  else
  {
    v7 = 1;
    if ( v16[0] == 227 )
      goto LABEL_8;
    if ( v16[0] > 0x21 )
      return 0;
  }
  v14 = 0x36D8D8360LL;
  if ( _bittest64(&v14, v16[0]) )
  {
    v7 = 2;
  }
  else
  {
    if ( v16[0] != 31 && v16[0] != 25 )
      return 0;
    v7 = 3;
  }
LABEL_8:
  v8 = *(_QWORD *)(v4 + 24);
  v9 = *(_QWORD **)(v8 + 16);
  if ( *(_BYTE *)(*v9 + 8LL) )
    return 0;
  if ( v7 != *(_DWORD *)(v8 + 12) - 1 )
    return 0;
  v10 = v9[1];
  v11 = sub_15E0530(v4);
  if ( v10 != sub_16471D0(v11, 0) || *(_BYTE *)(a1 + 16) != 78 )
    return 0;
  return a1;
}
