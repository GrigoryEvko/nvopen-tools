// Function: sub_185B0A0
// Address: 0x185b0a0
//
__int64 __fastcall sub_185B0A0(__int64 a1)
{
  __int64 i; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi

  if ( ((*(_WORD *)(a1 + 18) >> 4) & 0x3FF) != 0 && ((*(_WORD *)(a1 + 18) >> 4) & 0x3FF) != 0x46 )
    return 0;
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v3 = sub_1648700(i);
      if ( *((_BYTE *)v3 + 16) == 78 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( !i )
        goto LABEL_10;
    }
    if ( (*((_WORD *)v3 + 9) & 3) == 2 )
      return 0;
  }
LABEL_10:
  v4 = *(_QWORD *)(a1 + 80);
  v5 = a1 + 72;
  if ( v4 != a1 + 72 )
  {
    while ( 1 )
    {
      v6 = v4 - 24;
      if ( !v4 )
        v6 = 0;
      if ( sub_157EBE0(v6) )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v5 == v4 )
        return 1;
    }
    return 0;
  }
  return 1;
}
