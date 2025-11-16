// Function: sub_349F6B0
// Address: 0x349f6b0
//
void __fastcall sub_349F6B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r12
  unsigned __int64 v5; // rdi

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 576;
  }
  else
  {
    v1 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v1 )
      return;
    v2 = *(_QWORD *)(a1 + 16);
    v3 = 72 * v1;
  }
  v4 = v2 + v3;
  do
  {
    while ( !*(_QWORD *)v2
         && (!*(_BYTE *)(v2 + 24) || !*(_QWORD *)(v2 + 8) && !*(_QWORD *)(v2 + 16))
         && !*(_QWORD *)(v2 + 32) )
    {
      v2 += 72;
      if ( v2 == v4 )
        return;
    }
    v5 = *(_QWORD *)(v2 + 40);
    if ( v5 != v2 + 56 )
      _libc_free(v5);
    v2 += 72;
  }
  while ( v2 != v4 );
}
