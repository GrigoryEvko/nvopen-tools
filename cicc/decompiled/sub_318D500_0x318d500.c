// Function: sub_318D500
// Address: 0x318d500
//
void __fastcall sub_318D500(unsigned __int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A34630;
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 48);
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
    v3 = *(_QWORD *)(a1 + 40);
  }
  if ( v3 != a1 + 56 )
    _libc_free(v3);
  v5 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23850;
  if ( v5 != a1 + 24 )
    j_j___libc_free_0(v5);
  j_j___libc_free_0(a1);
}
