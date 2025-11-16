// Function: sub_30FA8A0
// Address: 0x30fa8a0
//
void __fastcall sub_30FA8A0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi

  v2 = *(_BYTE *)(a1 + 544) == 0;
  *(_QWORD *)a1 = &unk_4A328D0;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 544) = 0;
    v4 = *(_QWORD *)(a1 + 496);
    if ( v4 != a1 + 512 )
      _libc_free(v4);
    sub_C7D6A0(*(_QWORD *)(a1 + 472), 8LL * *(unsigned int *)(a1 + 488), 8);
  }
  v3 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A1F3E0;
  if ( v3 )
    sub_B91220(a1 + 32, v3);
}
