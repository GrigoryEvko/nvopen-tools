// Function: sub_1605960
// Address: 0x1605960
//
void __fastcall sub_1605960(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v2 = *(unsigned __int64 **)(a1 + 16);
  v3 = &v2[*(unsigned int *)(a1 + 24)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    _libc_free(v4);
  }
  v5 = *(unsigned __int64 **)(a1 + 64);
  v6 = (unsigned __int64)&v5[2 * *(unsigned int *)(a1 + 72)];
  if ( v5 != (unsigned __int64 *)v6 )
  {
    do
    {
      v7 = *v5;
      v5 += 2;
      _libc_free(v7);
    }
    while ( (unsigned __int64 *)v6 != v5 );
    v6 = *(_QWORD *)(a1 + 64);
  }
  if ( v6 != a1 + 80 )
    _libc_free(v6);
  v8 = *(_QWORD *)(a1 + 16);
  if ( v8 != a1 + 32 )
    _libc_free(v8);
}
