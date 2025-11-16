// Function: sub_25D7E30
// Address: 0x25d7e30
//
__int64 __fastcall sub_25D7E30(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // r14
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  unsigned __int64 v16; // rdi

  v2 = *(_BYTE *)(a1 + 676) == 0;
  *(_QWORD *)a1 = off_4A1F338;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 656));
  v3 = *(unsigned int *)(a1 + 640);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 624);
    v5 = &v4[17 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = v4[13];
        while ( v6 )
        {
          sub_25D6BE0(*(_QWORD *)(v6 + 24));
          v7 = v6;
          v6 = *(_QWORD *)(v6 + 16);
          j_j___libc_free_0(v7);
        }
        v8 = v4[1];
        if ( (_QWORD *)v8 != v4 + 3 )
          _libc_free(v8);
      }
      v4 += 17;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 640);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 624), 136 * v3, 8);
  v9 = *(_QWORD **)(a1 + 576);
  while ( v9 )
  {
    v10 = (unsigned __int64)v9;
    v9 = (_QWORD *)*v9;
    j_j___libc_free_0(v10);
  }
  memset(*(void **)(a1 + 560), 0, 8LL * *(_QWORD *)(a1 + 568));
  v11 = *(_QWORD *)(a1 + 560);
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  if ( v11 != a1 + 608 )
    j_j___libc_free_0(v11);
  sub_25D7B50(a1 + 504);
  v12 = *(_QWORD *)(a1 + 504);
  if ( v12 != a1 + 552 )
    j_j___libc_free_0(v12);
  v13 = *(unsigned int *)(a1 + 496);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD *)(a1 + 480);
    v15 = v14 + 72 * v13;
    do
    {
      while ( *(_QWORD *)v14 == -8192 || *(_QWORD *)v14 == -4096 || *(_BYTE *)(v14 + 36) )
      {
        v14 += 72;
        if ( v15 == v14 )
          goto LABEL_26;
      }
      v16 = *(_QWORD *)(v14 + 16);
      v14 += 72;
      _libc_free(v16);
    }
    while ( v15 != v14 );
LABEL_26:
    v13 = *(unsigned int *)(a1 + 496);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 480), 72 * v13, 8);
  if ( !*(_BYTE *)(a1 + 212) )
    _libc_free(*(_QWORD *)(a1 + 192));
  return sub_BB9260(a1);
}
