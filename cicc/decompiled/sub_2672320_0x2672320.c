// Function: sub_2672320
// Address: 0x2672320
//
__int64 __fastcall sub_2672320(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = *(unsigned __int64 **)(a1 + 464);
  *(_QWORD *)a1 = off_4A20118;
  *(_QWORD *)(a1 + 88) = &unk_4A201C8;
  if ( v2 )
  {
    if ( (unsigned __int64 *)*v2 != v2 + 2 )
      _libc_free(*v2);
    j_j___libc_free_0((unsigned __int64)v2);
  }
  if ( !*(_BYTE *)(a1 + 500) )
    _libc_free(*(_QWORD *)(a1 + 480));
  v3 = *(_QWORD *)(a1 + 320);
  if ( v3 != a1 + 336 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 296), 8LL * *(unsigned int *)(a1 + 312), 8);
  v4 = *(unsigned int *)(a1 + 280);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 264);
    v6 = v5 + (v4 << 7);
    do
    {
      if ( *(_QWORD *)v5 != -4 && *(_QWORD *)v5 != -16 )
      {
        if ( !*(_BYTE *)(v5 + 92) )
          _libc_free(*(_QWORD *)(v5 + 72));
        if ( !*(_BYTE *)(v5 + 44) )
          _libc_free(*(_QWORD *)(v5 + 24));
      }
      v5 += 128;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 280);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 264), v4 << 7, 8);
  v7 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 232);
    v9 = v8 + (v7 << 7);
    while ( 1 )
    {
      if ( *(_QWORD *)v8 == -8192 || *(_QWORD *)v8 == -4096 )
        goto LABEL_22;
      if ( *(_BYTE *)(v8 + 92) )
      {
        if ( *(_BYTE *)(v8 + 44) )
          goto LABEL_22;
LABEL_27:
        v10 = *(_QWORD *)(v8 + 24);
        v8 += 128;
        _libc_free(v10);
        if ( v9 == v8 )
        {
LABEL_28:
          v7 = *(unsigned int *)(a1 + 248);
          break;
        }
      }
      else
      {
        _libc_free(*(_QWORD *)(v8 + 72));
        if ( !*(_BYTE *)(v8 + 44) )
          goto LABEL_27;
LABEL_22:
        v8 += 128;
        if ( v9 == v8 )
          goto LABEL_28;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 232), v7 << 7, 8);
  if ( !*(_BYTE *)(a1 + 188) )
    _libc_free(*(_QWORD *)(a1 + 168));
  if ( !*(_BYTE *)(a1 + 140) )
    _libc_free(*(_QWORD *)(a1 + 120));
  v11 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v11 != a1 + 56 )
    _libc_free(v11);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
