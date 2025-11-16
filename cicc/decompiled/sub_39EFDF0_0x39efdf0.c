// Function: sub_39EFDF0
// Address: 0x39efdf0
//
void __fastcall sub_39EFDF0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 *v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v1 = 0;
  v3 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v3 )
    v1 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v3 - 32);
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 264) + 480LL) )
    sub_16BD130(".bundle_unlock forbidden when bundling is disabled", 1u);
  if ( !sub_39EF7F0(a1) )
    sub_16BD130(".bundle_unlock without matching lock", 1u);
  if ( (*(_BYTE *)(v1 + 44) & 1) != 0 )
    sub_16BD130("Empty bundle-locked group is forbidden", 1u);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 264) + 484LL) & 1) != 0 )
  {
    v4 = *(__int64 **)(*(_QWORD *)(a1 + 328) + 8LL * *(unsigned int *)(a1 + 336) - 8);
    sub_38D7880(v1, 0);
    if ( !sub_39EF7F0(a1) )
    {
      v5 = sub_38D4BB0(a1, v4[7]);
      sub_39EFA40(a1, v5, (__int64)v4);
      --*(_DWORD *)(a1 + 336);
      v6 = v4[14];
      if ( (__int64 *)v6 != v4 + 16 )
        _libc_free(v6);
      v7 = v4[8];
      if ( (__int64 *)v7 != v4 + 10 )
        _libc_free(v7);
      nullsub_1930();
      j_j___libc_free_0((unsigned __int64)v4);
    }
    if ( *(_DWORD *)(v1 + 36) != 2 )
      *(_BYTE *)(sub_38D4BB0(a1, 0) + 48) = 0;
  }
  else
  {
    sub_38D7880(v1, 0);
  }
}
