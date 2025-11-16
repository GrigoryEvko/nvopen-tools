// Function: sub_1C13E30
// Address: 0x1c13e30
//
__int64 __fastcall sub_1C13E30(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // r13
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 + 137) == 0;
  *(_QWORD *)a1 = &unk_49F7608;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
    {
      sub_1633490(*(_QWORD ***)(a1 + 16));
      j_j___libc_free_0(v3, 736);
    }
  }
  if ( *(_BYTE *)(a1 + 136) )
  {
    v4 = *(_QWORD *)(a1 + 8);
    if ( v4 )
    {
      sub_16025D0(*(_QWORD **)(a1 + 8));
      j_j___libc_free_0(v4, 8);
    }
  }
  v5 = *(_QWORD *)(a1 + 144);
  if ( v5 )
  {
    v6 = *(unsigned __int64 **)(v5 + 16);
    v7 = &v6[*(unsigned int *)(v5 + 24)];
    while ( v7 != v6 )
    {
      v8 = *v6++;
      _libc_free(v8);
    }
    v9 = *(unsigned __int64 **)(v5 + 64);
    v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(v5 + 72)];
    if ( v9 != (unsigned __int64 *)v10 )
    {
      do
      {
        v11 = *v9;
        v9 += 2;
        _libc_free(v11);
      }
      while ( (unsigned __int64 *)v10 != v9 );
      v10 = *(_QWORD *)(v5 + 64);
    }
    if ( v10 != v5 + 80 )
      _libc_free(v10);
    v12 = *(_QWORD *)(v5 + 16);
    if ( v12 != v5 + 32 )
      _libc_free(v12);
    j_j___libc_free_0(v5, 104);
  }
  v13 = *(_QWORD *)(a1 + 80);
  result = a1 + 96;
  if ( v13 != a1 + 96 )
    return j_j___libc_free_0(v13, *(_QWORD *)(a1 + 96) + 1LL);
  return result;
}
