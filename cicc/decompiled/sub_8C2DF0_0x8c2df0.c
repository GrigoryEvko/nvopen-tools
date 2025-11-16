// Function: sub_8C2DF0
// Address: 0x8c2df0
//
_QWORD *__fastcall sub_8C2DF0(__int64 a1, __int64 a2)
{
  __int64 i; // rdx
  __int64 j; // rax
  __int64 **v5; // r8
  __int64 v6; // rdi
  __int64 *v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rcx
  __int64 *k; // rax
  __int64 v11; // rdx
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_QWORD *)(a2 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v5 = *(__int64 ***)(j + 168);
  v13[0] = j;
  v6 = *(_QWORD *)(i + 168);
  v7 = *v5;
  v8 = *(_QWORD **)v6;
  if ( *v5 && v8 )
  {
    do
    {
      if ( (v8[4] & 1) != 0 )
        *((_BYTE *)v7 + 32) |= 1u;
      v8 = (_QWORD *)*v8;
      v7 = (__int64 *)*v7;
    }
    while ( v8 && v7 );
  }
  if ( (*(_BYTE *)(v6 + 16) & 0x20) != 0 )
    *((_BYTE *)v5 + 16) |= 0x20u;
  v9 = *(_QWORD **)v6;
  if ( *(_QWORD *)v6 )
  {
    do
    {
      for ( k = (__int64 *)v9[7]; k; k = (__int64 *)*k )
      {
        while ( 1 )
        {
          if ( *((_BYTE *)k + 8) == 6 )
          {
            v11 = k[2];
            if ( *(_BYTE *)(v11 + 140) == 9 && (*(_BYTE *)(*(_QWORD *)(v11 + 168) + 109LL) & 0x20) != 0 )
              break;
          }
          k = (__int64 *)*k;
          if ( !k )
            goto LABEL_22;
        }
        *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v11 + 96LL) + 181LL) &= ~2u;
        *(_QWORD *)(*(_QWORD *)(v11 + 168) + 240LL) = 0;
      }
LABEL_22:
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
  }
  *(_BYTE *)(a2 + 192) = (*(_BYTE *)(a2 + 192) | *(_BYTE *)(a1 + 192)) & 1 | *(_BYTE *)(a2 + 192) & 0xFE;
  *(_BYTE *)(a2 + 193) = (*(_BYTE *)(a2 + 193) | *(_BYTE *)(a1 + 193)) & 0x40 | *(_BYTE *)(a2 + 193) & 0xBF;
  if ( (*(_BYTE *)(v6 + 20) & 1) != 0 && (*((_BYTE *)v5 + 20) & 1) == 0 )
  {
    sub_73EA10((const __m128i **)(a2 + 152), v13);
    *(_BYTE *)(*(_QWORD *)(v13[0] + 168) + 20LL) |= 1u;
  }
  if ( (*(_BYTE *)(a1 + 195) & 0x10) != 0 )
    *(_BYTE *)(a2 + 195) |= 0x10u;
  if ( (*(_BYTE *)(a1 + 196) & 0x40) != 0 )
    *(_BYTE *)(a2 + 196) |= 0x40u;
  if ( *(_DWORD *)(a1 + 160) && *(_DWORD *)(a2 + 160) )
    *(_BYTE *)(a2 + 203) = ((((*(_BYTE *)(a1 + 203) & 0x40) != 0) & (*(_BYTE *)(a2 + 203) >> 6)) << 6)
                         | *(_BYTE *)(a2 + 203) & 0xBF;
  return sub_8C2B90(a1, a2);
}
