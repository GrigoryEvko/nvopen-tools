// Function: sub_1E1D240
// Address: 0x1e1d240
//
void *__fastcall sub_1E1D240(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  v2 = *(unsigned int *)(a1 + 1840);
  *(_QWORD *)a1 = off_49FBC40;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 1824);
    v4 = v3 + 32 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 8);
          if ( v5 )
            break;
        }
        v3 += 32;
        if ( v4 == v3 )
          goto LABEL_7;
      }
      v6 = *(_QWORD *)(v3 + 24);
      v3 += 32;
      j_j___libc_free_0(v5, v6 - v5);
    }
    while ( v4 != v3 );
  }
LABEL_7:
  j___libc_free_0(*(_QWORD *)(a1 + 1824));
  v7 = *(unsigned __int64 **)(a1 + 1032);
  v8 = &v7[6 * *(unsigned int *)(a1 + 1040)];
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 6;
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        _libc_free(*v8);
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 1032);
  }
  if ( v8 != (unsigned __int64 *)(a1 + 1048) )
    _libc_free((unsigned __int64)v8);
  v9 = *(_QWORD *)(a1 + 984);
  if ( v9 != a1 + 1000 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 936);
  if ( v10 != a1 + 952 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 904);
  while ( v11 )
  {
    sub_1E1D070(*(_QWORD *)(v11 + 24));
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
    j_j___libc_free_0(v12, 40);
  }
  v13 = *(_QWORD *)(a1 + 744);
  if ( v13 != a1 + 760 )
    _libc_free(v13);
  j___libc_free_0(*(_QWORD *)(a1 + 720));
  v14 = *(_QWORD *)(a1 + 624);
  if ( v14 != a1 + 640 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 464);
  if ( v15 != a1 + 480 )
    _libc_free(v15);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
