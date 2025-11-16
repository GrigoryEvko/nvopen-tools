// Function: sub_250E960
// Address: 0x250e960
//
__int64 __fastcall sub_250E960(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned int v5; // eax
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  _QWORD *v16; // rax
  _QWORD *v17; // r13
  _QWORD *v18; // rbx
  __int64 v19; // rdi

  if ( *(_DWORD *)(a1 + 24) )
  {
    v16 = *(_QWORD **)(a1 + 16);
    v17 = &v16[2 * *(unsigned int *)(a1 + 32)];
    if ( v16 != v17 )
    {
      while ( 1 )
      {
        v18 = v16;
        if ( *v16 != -8192 && *v16 != -4096 )
          break;
        v16 += 2;
        if ( v17 == v16 )
          goto LABEL_2;
      }
      while ( v17 != v18 )
      {
        v19 = v18[1];
        v18 += 2;
        sub_250E890(v19);
        if ( v18 == v17 )
          break;
        while ( *v18 == -4096 || *v18 == -8192 )
        {
          v18 += 2;
          if ( v17 == v18 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  v2 = *(_QWORD *)(a1 + 216);
  v3 = v2 + 8LL * *(unsigned int *)(a1 + 232);
  if ( *(_DWORD *)(a1 + 224) && v3 != v2 )
  {
    while ( *(_QWORD *)v2 == -4096 || *(_QWORD *)v2 == -8192 )
    {
      v2 += 8;
      if ( v2 == v3 )
        goto LABEL_3;
    }
LABEL_40:
    if ( v2 != v3 )
    {
      if ( !*(_BYTE *)(*(_QWORD *)v2 + 28LL) )
        _libc_free(*(_QWORD *)(*(_QWORD *)v2 + 8LL));
      while ( 1 )
      {
        v2 += 8;
        if ( v2 == v3 )
          break;
        if ( *(_QWORD *)v2 != -8192 && *(_QWORD *)v2 != -4096 )
          goto LABEL_40;
      }
    }
  }
LABEL_3:
  v4 = *(_QWORD *)(a1 + 120);
  if ( v4 )
  {
    sub_C7D6A0(*(_QWORD *)(v4 + 208), 8LL * *(unsigned int *)(v4 + 224), 8);
    v5 = *(_DWORD *)(v4 + 192);
    if ( v5 )
    {
      v6 = *(_QWORD **)(v4 + 176);
      v7 = &v6[2 * v5];
      do
      {
        if ( *v6 != -8192 && *v6 != -4096 )
        {
          v8 = v6[1];
          if ( v8 )
          {
            sub_C7D6A0(*(_QWORD *)(v8 + 8), 8LL * *(unsigned int *)(v8 + 24), 8);
            j_j___libc_free_0(v8);
          }
        }
        v6 += 2;
      }
      while ( v7 != v6 );
      v5 = *(_DWORD *)(v4 + 192);
    }
    sub_C7D6A0(*(_QWORD *)(v4 + 176), 16LL * v5, 8);
    sub_C7D6A0(*(_QWORD *)(v4 + 144), 16LL * *(unsigned int *)(v4 + 160), 8);
    sub_C7D6A0(*(_QWORD *)(v4 + 112), 16LL * *(unsigned int *)(v4 + 128), 8);
    sub_A17130(v4 + 72);
    sub_A17130(v4 + 40);
    sub_A17130(v4 + 8);
  }
  v9 = *(_QWORD *)(a1 + 344);
  if ( v9 != a1 + 360 )
    j_j___libc_free_0(v9);
  if ( !*(_BYTE *)(a1 + 276) )
    _libc_free(*(_QWORD *)(a1 + 256));
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 8LL * *(unsigned int *)(a1 + 232), 8);
  v10 = *(_QWORD *)(a1 + 192);
  if ( v10 != a1 + 208 )
    _libc_free(v10);
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 8LL * *(unsigned int *)(a1 + 184), 8);
  v11 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a1 + 136);
    v13 = v12 + 48 * v11;
    while ( *(_QWORD *)v12 == -4096 )
    {
      if ( *(_DWORD *)(v12 + 8) == 100 )
      {
        v12 += 48;
        if ( v13 == v12 )
        {
LABEL_27:
          v11 = *(unsigned int *)(a1 + 152);
          goto LABEL_28;
        }
      }
      else
      {
LABEL_22:
        sub_C7D6A0(*(_QWORD *)(v12 + 24), 24LL * *(unsigned int *)(v12 + 40), 8);
LABEL_23:
        v12 += 48;
        if ( v13 == v12 )
          goto LABEL_27;
      }
    }
    if ( *(_QWORD *)v12 == -8192 && *(_DWORD *)(v12 + 8) == 101 )
      goto LABEL_23;
    goto LABEL_22;
  }
LABEL_28:
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 48 * v11, 8);
  v14 = *(_QWORD *)(a1 + 40);
  if ( v14 != a1 + 56 )
    _libc_free(v14);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
