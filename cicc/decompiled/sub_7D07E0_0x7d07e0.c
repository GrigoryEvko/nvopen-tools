// Function: sub_7D07E0
// Address: 0x7d07e0
//
__int64 __fastcall sub_7D07E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // r14
  __int64 v10; // rbx
  char v11; // al
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  if ( *(_BYTE *)(a2 + 80) == 17 )
  {
    v10 = *(_QWORD *)(a2 + 88);
    if ( !v10 )
      return a1;
    while ( 1 )
    {
      v11 = *(_BYTE *)(v10 + 83);
      if ( (v11 & 0x40) == 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        return a1;
    }
    if ( a1 )
    {
      v8 = a1;
      if ( *(_BYTE *)(a1 + 80) != 24 || *(_QWORD *)(a1 + 88) )
        goto LABEL_14;
      sub_879260(a1, v10, 0xFFFFFFFFLL);
      v10 = *(_QWORD *)(v10 + 8);
    }
    else
    {
      v15 = sub_87ED40(v10, a3, a4, a5, a6);
      v10 = *(_QWORD *)(v10 + 8);
      v8 = v15;
    }
    if ( !v10 )
      return v8;
    v11 = *(_BYTE *)(v10 + 83);
LABEL_14:
    v12 = a3 + 8;
    while ( 1 )
    {
      if ( (v11 & 0x40) == 0 && (!a1 || !sub_7D06D0(v8, v10, 1, a6)) )
      {
        v13 = sub_87ECE0(v10, v12, (unsigned int)dword_4F04C64);
        v8 = sub_887160(v13, v8, a4, a5);
      }
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        break;
      v11 = *(_BYTE *)(v10 + 83);
    }
    return v8;
  }
  if ( !a1 )
    return sub_87ED40(a2, a3, a4, a5, a6);
  if ( *(_BYTE *)(a1 + 80) == 24 && !*(_QWORD *)(a1 + 88) )
  {
    v8 = a1;
    sub_879260(a1, a2, 0xFFFFFFFFLL);
    return v8;
  }
  if ( sub_7D06D0(a1, a2, 1, a6) )
    return a1;
  v14 = sub_87ECE0(a2, a3 + 8, (unsigned int)dword_4F04C64);
  return sub_887160(v14, a1, a4, a5);
}
