// Function: sub_2305800
// Address: 0x2305800
//
void __fastcall sub_2305800(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi

  v1 = a1 + 24;
  v2 = 16LL * *(unsigned int *)(a1 + 80);
  *(_QWORD *)a1 = &unk_4A0B420;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), v2, 8);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 != v1 )
    _libc_free(v3);
}
