// Function: sub_828A00
// Address: 0x828a00
//
_BOOL8 __fastcall sub_828A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  char v6; // dl
  __int64 v7; // rbx
  char v8; // al
  char v9; // al
  _BOOL8 result; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rsi

  v5 = a1;
  v6 = *(_BYTE *)(a1 + 80);
  v7 = a2;
  v8 = *(_BYTE *)(a2 + 80);
  if ( v6 == 16 )
  {
    v15 = *(_QWORD **)(a1 + 88);
    if ( v8 == 16 )
    {
      v20 = v15[1];
      v21 = *(_QWORD *)(*(_QWORD *)(v7 + 88) + 8LL);
      if ( v20 != v21 )
      {
        v22 = *(_QWORD *)(v20 + 64);
        if ( *(_QWORD *)(v21 + 64) != v22 || !v22 )
          return 0;
      }
    }
    v5 = *v15;
    v6 = *(_BYTE *)(*v15 + 80LL);
  }
  if ( v6 == 24 )
    v5 = *(_QWORD *)(v5 + 88);
  if ( v8 == 16 )
  {
    v7 = **(_QWORD **)(v7 + 88);
    v8 = *(_BYTE *)(v7 + 80);
  }
  if ( v8 == 24 )
    v7 = *(_QWORD *)(v7 + 88);
  if ( v7 == v5 )
    return 1;
  v9 = *(_BYTE *)(v5 + 80);
  if ( v9 != *(_BYTE *)(v7 + 80) )
    return 0;
  v11 = *(_QWORD *)(v5 + 88);
  v12 = *(_QWORD *)(v7 + 88);
  if ( (unsigned __int8)(v9 - 10) <= 1u )
  {
    if ( v12 != v11 && (!*qword_4D03FD0 || !v11 || !v12 || !(unsigned int)sub_8C7EB0(v11, v12)) )
    {
      if ( !dword_4F077BC )
        return 0;
      v16 = *(_QWORD *)(v5 + 88);
      if ( (*(_BYTE *)(v16 + 88) & 0x70) != 0x30 )
        return 0;
      v17 = *(_QWORD *)(v7 + 88);
      if ( (*(_BYTE *)(v17 + 88) & 0x70) != 0x30 )
        return 0;
      v18 = *(_QWORD *)(v16 + 152);
      v19 = *(_QWORD *)(v17 + 152);
      if ( v18 != v19 )
        return (unsigned int)sub_8D97D0(v18, v19, 0, v17, a5) != 0;
    }
    return 1;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(v11 + 104) + 200LL);
  v14 = *(_QWORD *)(*(_QWORD *)(v12 + 104) + 200LL);
  result = 1;
  if ( v13 != v14 )
  {
    result = 0;
    if ( *qword_4D03FD0 )
    {
      if ( v13 && v14 )
        return (unsigned int)sub_8C7EB0(v13, v14) != 0;
      return 0;
    }
  }
  return result;
}
