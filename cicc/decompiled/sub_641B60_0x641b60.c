// Function: sub_641B60
// Address: 0x641b60
//
__int64 __fastcall sub_641B60(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char v8; // al
  unsigned int v9; // r12d
  char v10; // r14
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rax

  v8 = *(_BYTE *)(a3 + 80);
  v9 = a2 == 0 ? 11 : 20;
  if ( a5 )
  {
    v10 = *(_BYTE *)(a3 + 83) & 0x40;
    if ( v8 == 11 )
    {
      v13 = *(_QWORD *)(a3 + 88);
      if ( !*(_BYTE *)(v13 + 174) )
      {
        if ( *(_WORD *)(v13 + 176) )
          sub_685490(1581, a1 + 8, a3);
      }
    }
    result = sub_887500(v9, a1, 0, a3, a4);
    *(_BYTE *)(result + 83) |= 0x40u;
    if ( v10 )
      *(_BYTE *)(*(_QWORD *)a4 + 83LL) |= 0x40u;
  }
  else
  {
    if ( v8 == 11 )
    {
      v14 = *(_QWORD *)(a3 + 88);
      if ( !*(_BYTE *)(v14 + 174) )
      {
        if ( *(_WORD *)(v14 + 176) )
          sub_685490(1581, a1 + 8, a3);
      }
    }
    result = sub_887500(v9, a1, 0, a3, a4);
    if ( !a6 )
      *(_BYTE *)(*(_QWORD *)a4 + 83LL) &= ~0x40u;
  }
  return result;
}
