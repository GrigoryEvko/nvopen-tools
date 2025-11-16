// Function: sub_2B2F3A0
// Address: 0x2b2f3a0
//
void __fastcall sub_2B2F3A0(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rax

  v2 = *(unsigned __int64 **)(a1 + 240);
  v3 = &v2[10 * *(unsigned int *)(a1 + 248)];
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 10;
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        _libc_free(*v3);
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 240);
  }
  if ( v3 != (unsigned __int64 *)(a1 + 256) )
    _libc_free((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 208);
  if ( v4 != a1 + 224 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 144);
  if ( v5 != a1 + 160 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 112);
  if ( v6 != a1 + 128 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 96);
  if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
    sub_BD60C0((_QWORD *)(a1 + 80));
  if ( *(_QWORD *)a1 != a1 + 16 )
    _libc_free(*(_QWORD *)a1);
}
