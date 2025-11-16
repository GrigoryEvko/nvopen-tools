// Function: sub_318DA10
// Address: 0x318da10
//
void __fastcall sub_318DA10(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12

  *(_QWORD *)a1 = &unk_4A346C0;
  v2 = *(_QWORD *)(a1 + 104);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(unsigned __int64 **)(a1 + 8);
  v4 = &v3[9 * *(unsigned int *)(a1 + 16)];
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 9;
      if ( (unsigned __int64 *)*v4 != v4 + 2 )
        _libc_free(*v4);
    }
    while ( v3 != v4 );
    v4 = *(unsigned __int64 **)(a1 + 8);
  }
  if ( v4 != (unsigned __int64 *)(a1 + 24) )
    _libc_free((unsigned __int64)v4);
}
