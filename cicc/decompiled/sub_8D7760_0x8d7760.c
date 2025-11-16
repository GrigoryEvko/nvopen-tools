// Function: sub_8D7760
// Address: 0x8d7760
//
__int64 __fastcall sub_8D7760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  char v6; // al
  _BYTE *v8; // rax

  if ( (*(_BYTE *)(a1 + 206) & 8) != 0 )
    sub_5F90A0(a1, a2, a3, a4, a5);
  if ( (*(_WORD *)(a1 + 194) & 0x1006) == 0 )
  {
    v5 = *(_QWORD *)(a1 + 152);
    v6 = *(_BYTE *)(v5 + 140);
    if ( v6 == 7 )
    {
      v8 = *(_BYTE **)(*(_QWORD *)(v5 + 168) + 56LL);
      if ( v8 && (*v8 & 2) != 0 )
        sub_5F80E0(a1);
      sub_894C00(*(_QWORD *)a1);
      if ( *(_BYTE *)(v5 + 140) != 12 )
        return sub_8D76D0(v5);
    }
    else if ( v6 != 12 )
    {
      return sub_8D76D0(v5);
    }
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
    return sub_8D76D0(v5);
  }
  return 1;
}
