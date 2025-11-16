// Function: sub_1B93BB0
// Address: 0x1b93bb0
//
__int64 __fastcall sub_1B93BB0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rdi

  v2 = *(_QWORD *)(a1 + 576);
  if ( v2 != *(_QWORD *)(a1 + 568) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 408);
  if ( v3 != *(_QWORD *)(a1 + 400) )
    _libc_free(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 272));
  v4 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 240);
    v6 = v5 + 80 * v4;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v5 <= 0xFFFFFFFD )
        {
          v7 = *(_QWORD *)(v5 + 24);
          if ( v7 != *(_QWORD *)(v5 + 16) )
            break;
        }
        v5 += 80;
        if ( v6 == v5 )
          goto LABEL_11;
      }
      _libc_free(v7);
      v5 += 80;
    }
    while ( v6 != v5 );
  }
LABEL_11:
  j___libc_free_0(*(_QWORD *)(a1 + 240));
  v8 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 208);
    v10 = v9 + 80 * v8;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v9 <= 0xFFFFFFFD )
        {
          v11 = *(_QWORD *)(v9 + 24);
          if ( v11 != *(_QWORD *)(v9 + 16) )
            break;
        }
        v9 += 80;
        if ( v10 == v9 )
          goto LABEL_17;
      }
      _libc_free(v11);
      v9 += 80;
    }
    while ( v10 != v9 );
  }
LABEL_17:
  j___libc_free_0(*(_QWORD *)(a1 + 208));
  v12 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD *)(a1 + 176);
    v14 = v13 + 80 * v12;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v13 <= 0xFFFFFFFD )
        {
          v15 = *(_QWORD *)(v13 + 24);
          if ( v15 != *(_QWORD *)(v13 + 16) )
            break;
        }
        v13 += 80;
        if ( v14 == v13 )
          goto LABEL_23;
      }
      _libc_free(v15);
      v13 += 80;
    }
    while ( v14 != v13 );
  }
LABEL_23:
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  v16 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD *)(a1 + 144);
    v18 = v17 + 40 * v16;
    do
    {
      while ( *(_DWORD *)v17 > 0xFFFFFFFD )
      {
        v17 += 40;
        if ( v18 == v17 )
          goto LABEL_28;
      }
      v19 = *(_QWORD *)(v17 + 16);
      v17 += 40;
      j___libc_free_0(v19);
    }
    while ( v18 != v17 );
  }
LABEL_28:
  j___libc_free_0(*(_QWORD *)(a1 + 144));
  v20 = *(_QWORD *)(a1 + 80);
  if ( v20 != *(_QWORD *)(a1 + 72) )
    _libc_free(v20);
  v21 = *(_QWORD *)(a1 + 40);
  if ( v21 )
    j_j___libc_free_0(v21, *(_QWORD *)(a1 + 56) - v21);
  return j___libc_free_0(*(_QWORD *)(a1 + 16));
}
