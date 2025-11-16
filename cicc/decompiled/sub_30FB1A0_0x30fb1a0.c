// Function: sub_30FB1A0
// Address: 0x30fb1a0
//
void __fastcall sub_30FB1A0(unsigned __int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  void (__fastcall *v7)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v8; // rdi

  v2 = *(unsigned int *)(a1 + 352);
  *(_QWORD *)a1 = &unk_4A32870;
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 8 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), 8LL * *(unsigned int *)(a1 + 320), 8);
  if ( !*(_BYTE *)(a1 + 284) )
    _libc_free(*(_QWORD *)(a1 + 264));
  v3 = *(_QWORD *)(a1 + 216);
  while ( v3 )
  {
    sub_30FAC60(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  v5 = *(_QWORD *)(a1 + 136);
  while ( v5 )
  {
    sub_30FAE30(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6);
  }
  v7 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 104);
  if ( v7 )
    v7(a1 + 88, a1 + 88, 3);
  v8 = *(_QWORD *)(a1 + 80);
  if ( v8 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  sub_30CBA20((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
