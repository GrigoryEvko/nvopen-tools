// Function: sub_1421ED0
// Address: 0x1421ed0
//
__int64 __fastcall sub_1421ED0(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned int v6; // eax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // rdi
  unsigned int v10; // eax
  _QWORD *v11; // r12
  _QWORD *v12; // r14
  unsigned __int64 *v13; // r13
  unsigned __int64 *v14; // r15
  unsigned __int64 *v15; // rdi
  unsigned __int64 v16; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // r10
  _QWORD *v20; // r9
  __int64 v21; // r11
  __int64 i; // r8
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx

  if ( *(_DWORD *)(a1 + 72) )
  {
    v18 = *(_QWORD **)(a1 + 64);
    v19 = &v18[2 * *(unsigned int *)(a1 + 80)];
    if ( v18 != v19 )
    {
      while ( 1 )
      {
        v20 = v18;
        if ( *v18 != -8 && *v18 != -16 )
          break;
        v18 += 2;
        if ( v19 == v18 )
          goto LABEL_2;
      }
      while ( v19 != v20 )
      {
        v21 = v20[1];
        for ( i = *(_QWORD *)(v21 + 8); v21 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          v23 = 3LL * (*(_DWORD *)(i - 12) & 0xFFFFFFF);
          if ( (*(_BYTE *)(i - 9) & 0x40) != 0 )
          {
            v24 = *(_QWORD **)(i - 40);
            v25 = &v24[v23];
          }
          else
          {
            v25 = (_QWORD *)(i - 32);
            v24 = (_QWORD *)(i - 32 - v23 * 8);
          }
          for ( ; v25 != v24; v24 += 3 )
          {
            if ( *v24 )
            {
              v26 = v24[1];
              v27 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v27 = v26;
              if ( v26 )
                *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
            }
            *v24 = 0;
          }
        }
        v20 += 2;
        if ( v20 == v19 )
          break;
        while ( *v20 == -16 || *v20 == -8 )
        {
          v20 += 2;
          if ( v19 == v20 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  v2 = *(_QWORD **)(a1 + 328);
  if ( v2 )
  {
    v3 = v2[265];
    *v2 = &unk_49EB390;
    j___libc_free_0(v3);
    v4 = v2[6];
    if ( (_QWORD *)v4 != v2 + 8 )
      _libc_free(v4);
    j_j___libc_free_0(v2, 2144);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 304));
  v5 = *(_QWORD *)(a1 + 144);
  if ( v5 != *(_QWORD *)(a1 + 136) )
    _libc_free(v5);
  if ( *(_QWORD *)(a1 + 120) )
    sub_164BEC0();
  v6 = *(_DWORD *)(a1 + 112);
  if ( v6 )
  {
    v7 = *(_QWORD **)(a1 + 96);
    v8 = &v7[2 * v6];
    do
    {
      if ( *v7 != -8 && *v7 != -16 )
      {
        v9 = v7[1];
        if ( v9 )
          j_j___libc_free_0(v9, 16);
      }
      v7 += 2;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 96));
  v10 = *(_DWORD *)(a1 + 80);
  if ( v10 )
  {
    v11 = *(_QWORD **)(a1 + 64);
    v12 = &v11[2 * v10];
    do
    {
      if ( *v11 != -8 && *v11 != -16 )
      {
        v13 = (unsigned __int64 *)v11[1];
        if ( v13 )
        {
          v14 = (unsigned __int64 *)v13[1];
          while ( v13 != v14 )
          {
            v15 = v14;
            v14 = (unsigned __int64 *)v14[1];
            v16 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
            *v14 = v16 | *v14 & 7;
            *(_QWORD *)(v16 + 8) = v14;
            *v15 &= 7u;
            v15[1] = 0;
            sub_164BEC0();
          }
          j_j___libc_free_0(v13, 16);
        }
      }
      v11 += 2;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 64));
  return j___libc_free_0(*(_QWORD *)(a1 + 32));
}
