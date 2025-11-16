// Function: sub_314D410
// Address: 0x314d410
//
void __fastcall sub_314D410(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi

  if ( (*(_BYTE *)(a1 + 44) & 4) != 0 )
  {
    v2 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v2 )
    {
      v3 = 8 * v2;
      v4 = 0;
      do
      {
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + v4);
        if ( v5 )
          j_j___libc_free_0_0(v5);
        v4 += 8;
      }
      while ( v3 != v4 );
    }
  }
  v6 = *(_QWORD *)(a1 + 80);
  if ( v6 )
  {
    sub_314D410(*(_QWORD *)(a1 + 80));
    j_j___libc_free_0(v6);
  }
  v7 = *(_QWORD *)(a1 + 48);
  if ( v7 != a1 + 64 )
    _libc_free(v7);
}
