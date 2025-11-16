// Function: sub_7E4690
// Address: 0x7e4690
//
void __fastcall sub_7E4690(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  char i; // al
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 j; // rdi

  if ( *(_BYTE *)(a1 + 173) == 10 )
  {
    v3 = *(_QWORD *)(a1 + 128);
    for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
      v3 = *(_QWORD *)(v3 + 160);
    if ( (unsigned __int8)(i - 9) <= 2u )
    {
      sub_7E3EE0(v3);
      sub_7DFA70((_QWORD *)a1, a2, v5, v6, v7);
      if ( (*(_BYTE *)(a1 + 171) & 0x12) == 2 )
      {
        for ( j = *(_QWORD *)(a1 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        sub_806430(j, 0, 0, a1, 0);
      }
    }
    *(_BYTE *)(a1 + 171) |= 8u;
  }
}
