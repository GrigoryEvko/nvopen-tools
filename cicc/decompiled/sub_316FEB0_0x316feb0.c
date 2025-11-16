// Function: sub_316FEB0
// Address: 0x316feb0
//
void __fastcall sub_316FEB0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v5; // r13
  bool v6; // cc
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdi

  if ( *(_BYTE *)(a1 + 676) )
  {
    if ( *(_BYTE *)(a1 + 628) )
      goto LABEL_3;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 656));
    if ( *(_BYTE *)(a1 + 628) )
      goto LABEL_3;
  }
  _libc_free(*(_QWORD *)(a1 + 608));
LABEL_3:
  v2 = *(_QWORD *)(a1 + 536);
  if ( v2 != a1 + 552 )
    _libc_free(v2);
  if ( *(_BYTE *)(a1 + 516) )
  {
    if ( *(_BYTE *)(a1 + 452) )
      goto LABEL_7;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 496));
    if ( *(_BYTE *)(a1 + 452) )
      goto LABEL_7;
  }
  _libc_free(*(_QWORD *)(a1 + 432));
LABEL_7:
  v3 = *(unsigned int *)(a1 + 416);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 400);
    v5 = v4 + 32 * v3;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 )
        {
          if ( *(_BYTE *)(v4 + 24) )
          {
            v6 = *(_DWORD *)(v4 + 16) <= 0x40u;
            *(_BYTE *)(v4 + 24) = 0;
            if ( !v6 )
            {
              v7 = *(_QWORD *)(v4 + 8);
              if ( v7 )
                break;
            }
          }
        }
        v4 += 32;
        if ( v5 == v4 )
          goto LABEL_16;
      }
      j_j___libc_free_0_0(v7);
      v4 += 32;
    }
    while ( v5 != v4 );
LABEL_16:
    v3 = *(unsigned int *)(a1 + 416);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 400), 32 * v3, 8);
  if ( *(_DWORD *)(a1 + 360) > 0x40u )
  {
    v8 = *(_QWORD *)(a1 + 352);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  if ( !*(_BYTE *)(a1 + 268) )
    _libc_free(*(_QWORD *)(a1 + 248));
  v9 = *(_QWORD *)(a1 + 32);
  v10 = v9 + 24LL * *(unsigned int *)(a1 + 40);
  if ( v9 != v10 )
  {
    do
    {
      v10 -= 24LL;
      if ( *(_DWORD *)(v10 + 16) > 0x40u )
      {
        v11 = *(_QWORD *)(v10 + 8);
        if ( v11 )
          j_j___libc_free_0_0(v11);
      }
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 32);
  }
  if ( v10 != a1 + 48 )
    _libc_free(v10);
}
