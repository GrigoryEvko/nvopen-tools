// Function: sub_1B7F330
// Address: 0x1b7f330
//
__int64 __fastcall sub_1B7F330(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int8 v7; // al
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rdx

  v2 = sub_1B7CA20(a2);
  v3 = sub_1649C60(v2);
  v4 = **(_QWORD **)(*(_QWORD *)v2 + 16LL);
  v5 = **(_QWORD **)(*(_QWORD *)v3 + 16LL);
  v6 = *(unsigned __int8 *)(v4 + 8);
  if ( ((unsigned __int8)v6 <= 0xFu && (v9 = 35454, _bittest64(&v9, v6))
     || ((unsigned int)(v6 - 13) <= 1 || (_DWORD)v6 == 16) && sub_16435F0(**(_QWORD **)(*(_QWORD *)v2 + 16LL), 0))
    && ((v10 = *(unsigned __int8 *)(v5 + 8), (unsigned __int8)v10 <= 0xFu) && (v12 = 35454, _bittest64(&v12, v10))
     || ((unsigned int)(v10 - 13) <= 1 || (_DWORD)v10 == 16) && sub_16435F0(v5, 0)) )
  {
    v11 = sub_127FA20(*(_QWORD *)(a1 + 40), v4);
    if ( (unsigned __int64)(sub_127FA20(*(_QWORD *)(a1 + 40), v5) + 7) >> 3 == (unsigned __int64)(v11 + 7) >> 3 )
      v2 = v3;
    v7 = *(_BYTE *)(v2 + 16);
    if ( v7 <= 0x17u )
      goto LABEL_17;
  }
  else
  {
    v7 = *(_BYTE *)(v2 + 16);
    if ( v7 <= 0x17u )
    {
LABEL_17:
      if ( v7 != 5 )
        return 0;
      if ( *(_WORD *)(v2 + 18) != 32 )
        return 0;
      return v2;
    }
  }
  if ( v7 != 56 )
    return 0;
  return v2;
}
