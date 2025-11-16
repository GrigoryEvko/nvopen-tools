// Function: sub_2548810
// Address: 0x2548810
//
void __fastcall sub_2548810(unsigned __int64 a1)
{
  __int64 v1; // rbx
  unsigned __int64 v2; // r12
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v5; // rbx

  v1 = *(_QWORD *)(a1 + 56);
  v2 = v1 + 16LL * *(unsigned int *)(a1 + 64);
  *(_QWORD *)a1 = &unk_4A170B8;
  if ( v1 != v2 )
  {
    do
    {
      v2 -= 16LL;
      if ( *(_DWORD *)(v2 + 8) > 0x40u && *(_QWORD *)v2 )
        j_j___libc_free_0_0(*(_QWORD *)v2);
    }
    while ( v1 != v2 );
    v2 = *(_QWORD *)(a1 + 56);
  }
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 48);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 32);
    v5 = v4 + 16 * v3;
    do
    {
      if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
        j_j___libc_free_0_0(*(_QWORD *)v4);
      v4 += 16;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 48);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 16 * v3, 8);
  j_j___libc_free_0(a1);
}
