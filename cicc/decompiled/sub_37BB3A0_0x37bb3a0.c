// Function: sub_37BB3A0
// Address: 0x37bb3a0
//
__int64 __fastcall sub_37BB3A0(__int64 a1, unsigned int a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  char *v5; // rdi

  v3 = sub_E922F0(*(_QWORD **)(a1 + 16), a2);
  v5 = &v3[2 * v4];
  if ( v3 == v5 )
    return 0;
  while ( (*(_QWORD *)(*(_QWORD *)(a1 + 56) + 8 * ((unsigned __int64)*(unsigned __int16 *)v3 >> 6))
         & (1LL << *(_WORD *)v3)) == 0 )
  {
    v3 += 2;
    if ( v5 == v3 )
      return 0;
  }
  return 1;
}
