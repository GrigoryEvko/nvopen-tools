// Function: sub_7DA4E0
// Address: 0x7da4e0
//
__int64 __fastcall sub_7DA4E0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 v8; // rbx
  __int64 i; // rbx
  __int64 j; // rbx
  __int64 k; // rbx
  int v12; // eax
  __int64 *v13; // rax
  __int64 m; // rbx
  _QWORD *n; // rbx
  __int64 *ii; // rbx
  char v17; // al
  __int64 v19; // r13
  _BYTE v20[144]; // [rsp+0h] [rbp-90h] BYREF

  v1 = a1;
  sub_7E18E0(v20, a1, 0);
  v6 = qword_4F04C50;
  v7 = *(_BYTE *)(a1 + 28);
  *(_BYTE *)(a1 - 8) |= 8u;
  if ( v7 == 17 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    for ( qword_4F04C50 = a1; v8; v8 = *(_QWORD *)(v8 + 112) )
      sub_7D8E20(v8);
  }
  else if ( v7 > 0x11u || (v7 & 0xFD) != 0 )
  {
    sub_721090();
  }
  for ( i = *(_QWORD *)(a1 + 120); i; i = *(_QWORD *)(i + 112) )
    sub_7D8E20(i);
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
    sub_7D8E20(j);
  for ( k = *(_QWORD *)(a1 + 104); k; k = *(_QWORD *)(k + 112) )
  {
    while ( 1 )
    {
      if ( !unk_4F072F3 )
        goto LABEL_15;
      sub_7E2D70(k);
      v12 = *(unsigned __int8 *)(k + 140);
      v2 = (unsigned int)(v12 - 10);
      if ( (unsigned __int8)(v12 - 10) <= 1u )
        break;
LABEL_16:
      if ( (_BYTE)v12 != 12 || !*(_QWORD *)(k + 8) || *(char *)(k + 186) >= 0 && !(unsigned int)sub_8D2B50(k) )
        goto LABEL_13;
      v13 = *(__int64 **)(k + 104);
      if ( !v13 )
        goto LABEL_13;
      do
      {
        if ( *((_BYTE *)v13 + 8) == 51 )
          *((_BYTE *)v13 + 8) = 0;
        v13 = (__int64 *)*v13;
      }
      while ( v13 );
      k = *(_QWORD *)(k + 112);
      if ( !k )
        goto LABEL_24;
    }
    v19 = *(_QWORD *)(k + 160);
    if ( v19 )
    {
      do
      {
        sub_7E2D70(v19);
        v19 = *(_QWORD *)(v19 + 112);
      }
      while ( v19 );
LABEL_15:
      LOBYTE(v12) = *(_BYTE *)(k + 140);
      goto LABEL_16;
    }
LABEL_13:
    ;
  }
LABEL_24:
  for ( m = *(_QWORD *)(a1 + 144); m; m = *(_QWORD *)(m + 112) )
  {
    while ( 1 )
    {
      sub_7E2D70(m);
      if ( (*(_BYTE *)(m + 201) & 1) != 0 )
        break;
      m = *(_QWORD *)(m + 112);
      if ( !m )
        goto LABEL_29;
    }
    sub_807D50(m);
  }
LABEL_29:
  for ( n = *(_QWORD **)(a1 + 160); n; n = (_QWORD *)*n )
    sub_7DA4E0(n);
  for ( ii = *(__int64 **)(a1 + 200); ii; ii = (__int64 *)*ii )
  {
    v1 = (__int64)(ii + 3);
    sub_7D8DC0(*((_BYTE *)ii + 16), (__m128i **)ii + 3);
  }
  v17 = *(_BYTE *)(a1 + 28);
  if ( v17 == 17 )
  {
    sub_7DA050(*(const __m128i **)(a1 + 80), v1, v2, v3, v4, v5);
    sub_7FAF20(*(_QWORD *)(a1 + 80));
    if ( unk_4D04380 && *(char *)(*(_QWORD *)(a1 + 32) + 192LL) < 0 )
      sub_76FD50((_QWORD *)a1);
  }
  else if ( !v17 && unk_4D04380 )
  {
    sub_76FE50();
  }
  qword_4F04C50 = v6;
  return sub_7E1AA0();
}
