// Function: sub_7D7B30
// Address: 0x7d7b30
//
__int64 __fastcall sub_7D7B30(__int64 a1)
{
  _BYTE *v1; // r12
  __int64 i; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 *v5; // rax
  char j; // al
  _BYTE *v7; // rax

  v1 = (_BYTE *)a1;
  for ( i = *(_QWORD *)a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
  {
    j = *(_BYTE *)(i + 140);
  }
  else
  {
    v3 = sub_7E7CB0(i);
    v4 = sub_7E2BE0(v3, a1);
    v5 = (__int64 *)sub_73E230(v3, a1);
    v1 = sub_73DF90(v4, v5);
    for ( j = *(_BYTE *)(i + 140); j == 12; j = *(_BYTE *)(i + 140) )
      i = *(_QWORD *)(i + 160);
  }
  if ( j == 5 )
    i = sub_7D7990(*(_BYTE *)(i + 160));
  v7 = sub_73DE50((__int64)v1, *(_QWORD *)(i + 160));
  return sub_7E2230(v7);
}
