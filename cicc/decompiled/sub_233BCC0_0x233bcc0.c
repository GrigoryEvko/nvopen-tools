// Function: sub_233BCC0
// Address: 0x233bcc0
//
void __fastcall sub_233BCC0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rbx
  unsigned __int64 v7; // rdi

  v2 = a1 + 2144;
  v3 = *(_QWORD *)(a1 + 2128);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 2120);
  v5 = *(_QWORD *)(a1 + 2104);
  v6 = a1 + 16;
  sub_C7D6A0(v5, 8 * v4, 8);
  sub_C7D6A0(*(_QWORD *)(v6 + 2056), 16LL * *(unsigned int *)(v6 + 2072), 8);
  v7 = *(_QWORD *)(v6 - 16);
  if ( v7 != v6 )
    _libc_free(v7);
}
