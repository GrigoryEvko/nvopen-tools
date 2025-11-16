// Function: sub_35E52F0
// Address: 0x35e52f0
//
void __fastcall sub_35E52F0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi

  *(_QWORD *)a1 = &unk_4A3ADC8;
  v2 = *(_QWORD *)(a1 + 272);
  if ( v2 != a1 + 288 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 248), 16LL * *(unsigned int *)(a1 + 264), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
  v3 = *(_QWORD *)(a1 + 144);
  if ( v3 != a1 + 160 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 80);
  if ( v4 != a1 + 96 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 72);
  if ( v5 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  j_j___libc_free_0(a1);
}
