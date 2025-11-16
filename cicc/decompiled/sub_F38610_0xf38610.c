// Function: sub_F38610
// Address: 0xf38610
//
void __fastcall sub_F38610(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 v6; // rdi

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 384;
  }
  else
  {
    v2 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v2 )
      return;
    v3 = *(_QWORD *)(a1 + 16);
    v4 = 96 * v2;
  }
  v5 = v3 + v4;
  do
  {
    while ( !*(_QWORD *)v3
         && (!*(_BYTE *)(v3 + 24) || !*(_QWORD *)(v3 + 8) && !*(_QWORD *)(v3 + 16))
         && !*(_QWORD *)(v3 + 32) )
    {
      v3 += 96;
      if ( v3 == v5 )
        return;
    }
    v6 = *(_QWORD *)(v3 + 40);
    if ( v6 != v3 + 56 )
      _libc_free(v6, a2);
    v3 += 96;
  }
  while ( v3 != v5 );
}
