// Function: sub_28CB820
// Address: 0x28cb820
//
void __fastcall sub_28CB820(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r12

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = v2 + 56 * v1;
    do
    {
      if ( *(_QWORD *)v2 != -8 && *(_QWORD *)v2 != 0x7FFFFFFF0LL && !*(_BYTE *)(v2 + 36) )
        _libc_free(*(_QWORD *)(v2 + 16));
      v2 += 56;
    }
    while ( v3 != v2 );
  }
}
