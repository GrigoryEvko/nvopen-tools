// Function: sub_87CD50
// Address: 0x87cd50
//
__int64 __fastcall sub_87CD50(
        __int64 a1,
        __int64 a2,
        FILE *a3,
        int a4,
        int a5,
        __int64 a6,
        unsigned int a7,
        _DWORD *a8)
{
  __int64 v8; // r10
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // rax

  v8 = a2;
  if ( a8 )
    *a8 = 0;
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  if ( !v10 )
    return 0;
  v11 = *(_QWORD *)(v10 + 24);
  if ( !v11 )
  {
    if ( (*(_BYTE *)(a1 + 180) & 0x10) == 0 || qword_4F07788 <= 0x577 )
      return 0;
    if ( a8 )
    {
      if ( !sub_67D3C0((int *)0x658, 7, a3) )
        return 0;
      *a8 = 1;
    }
    else
    {
      sub_685260(7u, 0x658u, a3, a1);
    }
    return v11;
  }
  if ( !a2 )
  {
    if ( !a5 )
      goto LABEL_9;
LABEL_17:
    v15 = *(_QWORD *)(v11 + 88);
    if ( (*(_BYTE *)(v15 + 194) & 8) == 0 || (*(_BYTE *)(v15 + 206) & 0x10) != 0 )
      sub_8769C0(v11, a3, v8, a4, 1, a6, a7, 0, a8);
    else
      sub_8769C0(v11, a3, v8, a4, 0, a6, a7, 0, a8);
    v12 = *(_QWORD *)(v10 + 24);
    if ( !v12 )
      return 0;
    v11 = *(_QWORD *)(v11 + 88);
    goto LABEL_10;
  }
  while ( *(_BYTE *)(v8 + 140) == 12 )
    v8 = *(_QWORD *)(v8 + 160);
  if ( a5 )
    goto LABEL_17;
LABEL_9:
  sub_8769C0(*(_QWORD *)(v10 + 24), a3, v8, a4, 0, a6, a7, 0, a8);
  v12 = *(_QWORD *)(v10 + 24);
  v11 = *(_QWORD *)(v11 + 88);
  if ( !v12 )
    return v11;
LABEL_10:
  v13 = *(_QWORD *)(v12 + 88);
  if ( (*(_BYTE *)(v13 + 194) & 8) == 0 || (*(_BYTE *)(v13 + 206) & 0x10) != 0 || !a5 )
    return v11;
  return 0;
}
