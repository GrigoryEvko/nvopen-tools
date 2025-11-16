// Function: sub_72FDF0
// Address: 0x72fdf0
//
__int64 __fastcall sub_72FDF0(__int64 a1, const __m128i *a2)
{
  __int64 i; // rbx
  unsigned int v3; // r12d
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  _BYTE *v9; // r15

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !(unsigned int)sub_8D32E0(i) )
  {
    if ( *(_BYTE *)(i + 140) == 20 )
    {
      sub_724C70((__int64)a2, 15);
      a2[8].m128i_i64[0] = a1;
      v3 = 1;
      a2[11].m128i_i8[0] = 0;
      a2[11].m128i_i64[1] = 0;
      return v3;
    }
    if ( (unsigned int)sub_8D3350(i) )
    {
      v3 = 1;
      sub_72BB40(a1, a2);
      return v3;
    }
    v5 = *(_BYTE *)(i + 140);
    if ( v5 == 12 )
    {
      v6 = i;
      do
      {
        v6 = *(_QWORD *)(v6 + 160);
        v5 = *(_BYTE *)(v6 + 140);
      }
      while ( v5 == 12 );
    }
    if ( !v5 )
    {
      v3 = 1;
      sub_72C970((__int64)a2);
      return v3;
    }
    if ( (unsigned int)sub_8D3BB0(i)
      || (unsigned int)sub_8D2B80(i)
      || (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u
      && (v7 = *(_QWORD *)(*(_QWORD *)i + 96LL), (*(_BYTE *)(v7 + 176) & 1) == 0)
      && (*(_QWORD *)(v7 + 16) || !*(_QWORD *)(v7 + 8)) )
    {
      sub_724C70((__int64)a2, 10);
      a2[8].m128i_i64[0] = a1;
      v3 = 1;
      if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
      {
        v8 = **(_QWORD **)(i + 168);
        if ( v8 )
        {
          do
          {
            if ( (*(_BYTE *)(v8 + 96) & 1) != 0 )
            {
              v9 = sub_724D50(10);
              v3 = sub_72FDF0(*(_QWORD *)(v8 + 40), v9);
              if ( !v3 )
                goto LABEL_28;
              *(_WORD *)(v9 + 171) |= 0x180u;
              sub_72A690((__int64)v9, (__int64)a2, v8, 0);
            }
            v8 = *(_QWORD *)v8;
          }
          while ( v8 );
          v3 = 1;
LABEL_28:
          if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 2u )
            goto LABEL_30;
        }
        else
        {
          v3 = 1;
        }
        if ( !sub_72FD90(*(_QWORD *)(i + 160), 3) )
          return v3;
      }
LABEL_30:
      a2[10].m128i_i8[10] |= 0x60u;
      return v3;
    }
  }
  return 0;
}
