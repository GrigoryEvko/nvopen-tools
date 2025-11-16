// Function: sub_2509740
// Address: 0x2509740
//
unsigned __int64 __fastcall sub_2509740(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax

  v1 = *a1 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*a1 & 3LL) == 3 )
    v1 = *(_QWORD *)(v1 + 24);
  v2 = *(_BYTE *)v1;
  if ( *(_BYTE *)v1 > 0x1Cu )
    return v1;
  if ( v2 == 22 )
  {
    if ( !sub_B2FC80(*(_QWORD *)(v1 + 24)) )
    {
      v6 = *(_QWORD *)(*(_QWORD *)(v1 + 24) + 80LL);
      if ( v6 )
      {
        v7 = *(_QWORD *)(v6 + 32);
        v1 = v7 - 24;
        if ( v7 )
          return v1;
        return 0;
      }
LABEL_16:
      BUG();
    }
    v2 = *(_BYTE *)v1;
  }
  if ( v2 || sub_B2FC80(v1) )
    return 0;
  v4 = *(_QWORD *)(v1 + 80);
  if ( !v4 )
    goto LABEL_16;
  v5 = *(_QWORD *)(v4 + 32);
  if ( v5 )
    return v5 - 24;
  return 0;
}
