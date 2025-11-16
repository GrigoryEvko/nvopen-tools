// Function: sub_FF6D40
// Address: 0xff6d40
//
__int64 __fastcall sub_FF6D40(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rax
  unsigned int v5; // edx
  __int64 v6; // rax
  _DWORD *v7; // r9

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_20;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_20:
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v4 = *(_QWORD *)(v2 - 120);
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v5 = *(_WORD *)(v4 + 2) & 0x3F;
  if ( v5 - 32 > 1 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 - 64) + 8LL) + 8LL) != 14 )
    return 0;
  v6 = qword_4F8E710;
  if ( !qword_4F8E710 )
    return 0;
  v7 = &unk_4F8E708;
  do
  {
    if ( v5 > *(_DWORD *)(v6 + 32) )
    {
      v6 = *(_QWORD *)(v6 + 24);
    }
    else
    {
      v7 = (_DWORD *)v6;
      v6 = *(_QWORD *)(v6 + 16);
    }
  }
  while ( v6 );
  if ( v7 == (_DWORD *)&unk_4F8E708 || v5 < v7[8] )
    return 0;
  sub_FF6650(a1, a2, (__int64)(v7 + 10));
  return 1;
}
