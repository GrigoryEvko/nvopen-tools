// Function: sub_31B6D40
// Address: 0x31b6d40
//
void __fastcall sub_31B6D40(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)a1 = &unk_4A34940;
  *(_QWORD *)(a1 + 40) = &unk_4A34630;
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 88);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 80);
  }
  if ( v3 != a1 + 96 )
    _libc_free(v3);
  v5 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 40) = &unk_4A23850;
  if ( v5 != a1 + 64 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23850;
  if ( v6 != a1 + 24 )
    j_j___libc_free_0(v6);
}
