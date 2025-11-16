// Function: sub_2A0B590
// Address: 0x2a0b590
//
__int64 __fastcall sub_2A0B590(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax

  v1 = **(_QWORD **)(a1 + 32);
  v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == v1 + 48 )
    goto LABEL_25;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_25:
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 31 )
    BUG();
  v3 = *(_QWORD *)(v2 - 56);
  if ( *(_BYTE *)(a1 + 84) )
  {
    v4 = *(_QWORD **)(a1 + 64);
    v5 = &v4[*(unsigned int *)(a1 + 76)];
    if ( v4 == v5 )
      goto LABEL_11;
    while ( v3 != *v4 )
    {
      if ( v5 == ++v4 )
        goto LABEL_11;
    }
  }
  else if ( !sub_C8CA60(a1 + 56, *(_QWORD *)(v2 - 56)) )
  {
    goto LABEL_11;
  }
  v3 = *(_QWORD *)(v2 - 88);
LABEL_11:
  v7 = sub_AA5930(v1);
  while ( v6 != v7 )
  {
    v8 = *(_QWORD *)(v7 + 16);
    if ( !v8 )
      return 1;
    while ( v3 == *(_QWORD *)(*(_QWORD *)(v8 + 24) + 40LL) )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        return 1;
    }
    v9 = *(_QWORD *)(v7 + 32);
    if ( !v9 )
      BUG();
    v7 = 0;
    if ( *(_BYTE *)(v9 - 24) == 84 )
      v7 = v9 - 24;
  }
  return 0;
}
