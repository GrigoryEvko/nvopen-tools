// Function: sub_318DE50
// Address: 0x318de50
//
void __fastcall sub_318DE50(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdi

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8LL;
      if ( v4 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL))(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)a1;
  }
  if ( v3 != a1 + 16 )
    _libc_free(v3);
}
