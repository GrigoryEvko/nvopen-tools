// Function: sub_278E480
// Address: 0x278e480
//
void __fastcall sub_278E480(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = v2 + (v1 << 6);
    do
    {
      v4 = *(_QWORD *)(v2 + 16);
      if ( v4 != v2 + 32 )
        _libc_free(v4);
      v2 += 64;
    }
    while ( v3 != v2 );
  }
}
