// Function: sub_2E981F0
// Address: 0x2e981f0
//
void __fastcall sub_2E981F0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  v2 = *(unsigned int *)(a1 + 1448);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 1432);
    v4 = v3 + 40 * v2;
    do
    {
      if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
      {
        v5 = *(unsigned int *)(v3 + 32);
        if ( (_DWORD)v5 )
        {
          v6 = *(_QWORD *)(v3 + 16);
          v7 = v6 + 32 * v5;
          do
          {
            while ( 1 )
            {
              if ( *(_DWORD *)v6 <= 0xFFFFFFFD )
              {
                v8 = *(_QWORD *)(v6 + 8);
                if ( v8 )
                  break;
              }
              v6 += 32;
              if ( v7 == v6 )
                goto LABEL_11;
            }
            v6 += 32;
            j_j___libc_free_0(v8);
          }
          while ( v7 != v6 );
LABEL_11:
          v5 = *(unsigned int *)(v3 + 32);
        }
        sub_C7D6A0(*(_QWORD *)(v3 + 16), 32 * v5, 8);
      }
      v3 += 40;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 1448);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1432), 40 * v2, 8);
  v9 = *(unsigned __int64 **)(a1 + 640);
  v10 = &v9[6 * *(unsigned int *)(a1 + 648)];
  if ( v9 != v10 )
  {
    do
    {
      v10 -= 6;
      if ( (unsigned __int64 *)*v10 != v10 + 2 )
        _libc_free(*v10);
    }
    while ( v9 != v10 );
    v10 = *(unsigned __int64 **)(a1 + 640);
  }
  if ( v10 != (unsigned __int64 *)(a1 + 656) )
    _libc_free((unsigned __int64)v10);
  v11 = *(_QWORD *)(a1 + 592);
  if ( v11 != a1 + 608 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 544);
  if ( v12 != a1 + 560 )
    _libc_free(v12);
  if ( (*(_BYTE *)(a1 + 520) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 528), 4LL * *(unsigned int *)(a1 + 536), 4);
  v13 = *(unsigned int *)(a1 + 504);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 488);
    v15 = &v14[11 * v13];
    do
    {
      if ( *v14 != -4096 && *v14 != -8192 )
      {
        v16 = v14[1];
        if ( (_QWORD *)v16 != v14 + 3 )
          _libc_free(v16);
      }
      v14 += 11;
    }
    while ( v15 != v14 );
    v13 = *(unsigned int *)(a1 + 504);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 488), 88 * v13, 8);
  if ( (*(_BYTE *)(a1 + 408) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 416), 16LL * *(unsigned int *)(a1 + 424), 8);
  v17 = *(_QWORD *)(a1 + 248);
  if ( v17 != a1 + 264 )
    _libc_free(v17);
}
