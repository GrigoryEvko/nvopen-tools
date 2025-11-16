// Function: sub_5CB000
// Address: 0x5cb000
//
__int64 __fastcall sub_5CB000(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  char v3; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // r14
  char *v9; // rax

  v2 = *(_QWORD *)(a2 + 152);
  v3 = *(_BYTE *)(v2 + 140);
  if ( v3 == 12 )
  {
    v4 = *(_QWORD *)(a2 + 152);
    do
    {
      v4 = *(_QWORD *)(v4 + 160);
      v5 = *(_BYTE *)(v4 + 140);
    }
    while ( v5 == 12 );
    if ( !v5 )
      goto LABEL_5;
    do
      v2 = *(_QWORD *)(v2 + 160);
    while ( *(_BYTE *)(v2 + 140) == 12 );
  }
  else if ( !v3 )
  {
    goto LABEL_5;
  }
  if ( *(_QWORD *)(*(_QWORD *)(v2 + 168) + 40LL) )
  {
    v7 = a1 + 56;
    v9 = sub_5C79F0(a1);
    sub_684B10(1860, a1 + 56, v9);
    goto LABEL_11;
  }
LABEL_5:
  v6 = *(_QWORD *)(a2 + 256);
  if ( !v6 )
    v6 = sub_726210(a2);
  if ( (unsigned int)sub_5CAD90(a1, 0, (_WORD *)(v6 + 32)) )
    *(_BYTE *)(a2 + 207) |= 4u;
  v7 = a1 + 56;
  if ( *(_BYTE *)(a1 + 8) )
    *(_BYTE *)(a2 + 200) |= 8u;
LABEL_11:
  sub_8767A0(4, *(_QWORD *)a2, v7, 1);
  return a2;
}
