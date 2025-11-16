// Function: sub_26AB9A0
// Address: 0x26ab9a0
//
void __fastcall sub_26AB9A0(__int64 a1)
{
  _QWORD **v2; // r12
  _QWORD **v3; // r13
  _QWORD *v4; // rdi
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 *v7; // r14
  unsigned __int64 *v8; // r12
  unsigned __int64 *v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  sub_BD84D0(*(_QWORD *)(a1 + 8), *(_QWORD *)a1);
  sub_B2E860(*(_QWORD **)(a1 + 8));
  if ( !*(_BYTE *)(a1 + 96) )
  {
    v2 = *(_QWORD ***)(a1 + 16);
    v3 = &v2[2 * *(unsigned int *)(a1 + 24)];
    while ( v3 != v2 )
    {
      v4 = *v2;
      v2 += 2;
      sub_B2E860(v4);
    }
  }
  v5 = *(_QWORD *)(a1 + 136);
  if ( v5 )
  {
    sub_FDC110(*(__int64 **)(a1 + 136));
    j_j___libc_free_0(v5);
  }
  v6 = *(_QWORD *)(a1 + 128);
  if ( v6 )
  {
    v7 = *(unsigned __int64 **)v6;
    v8 = (unsigned __int64 *)(*(_QWORD *)v6 + 104LL * *(unsigned int *)(v6 + 8));
    if ( *(unsigned __int64 **)v6 != v8 )
    {
      do
      {
        v8 -= 13;
        if ( (unsigned __int64 *)*v8 != v8 + 2 )
          _libc_free(*v8);
      }
      while ( v7 != v8 );
      v8 = *(unsigned __int64 **)v6;
    }
    if ( v8 != (unsigned __int64 *)(v6 + 16) )
      _libc_free((unsigned __int64)v8);
    j_j___libc_free_0(v6);
  }
  v9 = *(unsigned __int64 **)(a1 + 120);
  if ( v9 )
  {
    v10 = v9[8];
    if ( (unsigned __int64 *)v10 != v9 + 10 )
      _libc_free(v10);
    if ( (unsigned __int64 *)*v9 != v9 + 2 )
      _libc_free(*v9);
    j_j___libc_free_0((unsigned __int64)v9);
  }
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 != a1 + 32 )
    _libc_free(v11);
}
