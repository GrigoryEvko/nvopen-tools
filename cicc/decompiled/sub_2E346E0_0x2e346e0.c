// Function: sub_2E346E0
// Address: 0x2e346e0
//
__int64 __fastcall sub_2E346E0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // r13
  unsigned __int64 *v4; // rbx
  unsigned __int64 v5; // rsi

  *(_QWORD *)a1 = &unk_4A288B0;
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 672LL) = 0;
  v2 = *(_QWORD *)(a1 + 56);
  v3 = (unsigned __int64 *)(v2 + 8LL * *(unsigned int *)(a1 + 64));
  if ( v3 != (unsigned __int64 *)v2 )
  {
    v4 = (unsigned __int64 *)v2;
    do
    {
      v5 = *v4++;
      sub_2E192D0(*(_QWORD *)(a1 + 16), v5, 0);
    }
    while ( v3 != v4 );
    v2 = *(_QWORD *)(a1 + 56);
  }
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 32), 8LL * *(unsigned int *)(a1 + 48), 8);
}
