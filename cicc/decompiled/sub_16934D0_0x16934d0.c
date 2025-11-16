// Function: sub_16934D0
// Address: 0x16934d0
//
void __fastcall sub_16934D0(__int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rdi

  v2 = *(_QWORD *)(a1 + 48);
  if ( (*(_BYTE *)(a1 + 44) & 2) != 0 )
  {
    v3 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD *)(v2 + v5);
        if ( v6 )
        {
          j_j___libc_free_0_0(v6);
          v2 = *(_QWORD *)(a1 + 48);
        }
        v5 += 8;
      }
      while ( v5 != v4 );
    }
  }
  if ( v2 != a1 + 64 )
    _libc_free(v2);
}
