// Function: sub_2FAF0B0
// Address: 0x2faf0b0
//
void __fastcall sub_2FAF0B0(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = 0;
  if ( v2 )
  {
    v3 = v2 + 112LL * *(_QWORD *)(v2 - 8);
    while ( v2 != v3 )
    {
      v3 -= 112;
      v4 = *(_QWORD *)(v3 + 24);
      if ( v4 != v3 + 40 )
        _libc_free(v4);
    }
    j_j_j___libc_free_0_0(v2 - 8);
  }
  *(_DWORD *)(a1 + 232) = 0;
}
