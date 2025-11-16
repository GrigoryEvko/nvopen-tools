// Function: sub_2E34780
// Address: 0x2e34780
//
void __fastcall sub_2E34780(unsigned __int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rsi

  *(_QWORD *)a1 = &unk_4A288B0;
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 672LL) = 0;
  v1 = *(unsigned __int64 **)(a1 + 56);
  v2 = (unsigned __int64)&v1[*(unsigned int *)(a1 + 64)];
  if ( v1 != (unsigned __int64 *)v2 )
  {
    do
    {
      v3 = *v1++;
      sub_2E192D0(*(_QWORD *)(a1 + 16), v3, 0);
    }
    while ( (unsigned __int64 *)v2 != v1 );
    v2 = *(_QWORD *)(a1 + 56);
  }
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 8LL * *(unsigned int *)(a1 + 48), 8);
  j_j___libc_free_0(a1);
}
