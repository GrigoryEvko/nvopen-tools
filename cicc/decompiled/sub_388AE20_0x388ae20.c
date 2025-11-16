// Function: sub_388AE20
// Address: 0x388ae20
//
void __fastcall sub_388AE20(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 i; // r12

  v2 = *(_QWORD *)(a1 + 152);
  if ( v2 )
    j_j___libc_free_0_0(v2);
  if ( *(void **)(a1 + 120) == sub_16982C0() )
  {
    v6 = *(_QWORD *)(a1 + 128);
    if ( v6 )
    {
      v7 = 32LL * *(_QWORD *)(v6 - 8);
      for ( i = v6 + v7; v6 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v6 - 8);
    }
  }
  else
  {
    sub_1698460(a1 + 120);
  }
  if ( *(_DWORD *)(a1 + 104) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 96);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  v4 = *(_QWORD *)(a1 + 64);
  if ( v4 != a1 + 80 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 32);
  if ( v5 != a1 + 48 )
    j_j___libc_free_0(v5);
}
