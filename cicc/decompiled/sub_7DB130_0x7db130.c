// Function: sub_7DB130
// Address: 0x7db130
//
__int64 __fastcall sub_7DB130(__m128i *a1, _QWORD *a2, __int64 *a3)
{
  __m128i *m; // r14
  __int64 v6; // rax
  __m128i *i; // rdi
  __m128i *v8; // r13
  __m128i *n; // rdi
  __m128i *v10; // rax
  __m128i *v11; // rax
  char v12; // al
  _QWORD *v13; // rax
  unsigned __int64 v14; // r13
  _BYTE *v15; // r15
  __int64 v16; // rax
  int v17; // r12d
  char v18; // al
  unsigned __int64 v19; // r15
  __m128i *k; // rdi
  __m128i *j; // rdi
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  _QWORD v24[7]; // [rsp+18h] [rbp-38h] BYREF

  m = a1;
  *a2 = 0;
  if ( a3 )
    *a3 = 0;
  if ( (unsigned int)sub_8D2FB0(a1) )
  {
    v6 = sub_8D46C0(a1);
    *a2 |= 8uLL;
    m = (__m128i *)v6;
    if ( (unsigned int)sub_7E1F40(v6) )
      goto LABEL_5;
  }
  else if ( (unsigned int)sub_7E1F40(a1) )
  {
LABEL_5:
    *a2 |= 0x40uLL;
    goto LABEL_6;
  }
  if ( (unsigned int)sub_7E1F90(m) )
  {
    *a2 |= 0x80uLL;
    if ( (unsigned int)sub_8D3D10(m) )
    {
      for ( i = m; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
        ;
      if ( (unsigned int)sub_8D76D0(i) )
      {
        while ( m[8].m128i_i8[12] == 12 )
          m = (__m128i *)m[10].m128i_i64[0];
        v11 = sub_73C240(m);
        *a2 |= 0x10uLL;
        m = v11;
      }
    }
  }
LABEL_6:
  if ( !(unsigned int)sub_8D2E30(m) || (unsigned int)sub_7E1E50(m) )
    return sub_7E1E20(m);
  v8 = (__m128i *)sub_8D46C0(m);
  if ( (unsigned int)sub_8D2E30(v8) && !(unsigned int)sub_7E1E50(v8) )
  {
    if ( a3 )
    {
      v13 = sub_72BA30(unk_4F06870);
      v23 = sub_7DB0A0((__int64)v13, 0, v24);
      do
      {
        v14 = 0;
        m = (__m128i *)sub_8D46C0(m);
        v17 = sub_8D2E30(m);
        if ( (m[8].m128i_i8[12] & 0xFB) == 8 )
        {
          v18 = sub_8D4C10(m, dword_4F077C4 != 2);
          v14 = (2 * v18) & 2;
          if ( (v18 & 2) != 0 )
            v14 |= 4u;
        }
        if ( (unsigned int)sub_7E1F40(m) )
        {
          v14 |= 0x40u;
        }
        else if ( (unsigned int)sub_7E1F90(m) )
        {
          v19 = v14;
          if ( !(unsigned int)sub_8D3D10(m) )
            goto LABEL_49;
          for ( j = m; j[8].m128i_i8[12] == 12; j = (__m128i *)j[10].m128i_i64[0] )
            ;
          if ( (unsigned int)sub_8D76D0(j) )
          {
            while ( m[8].m128i_i8[12] == 12 )
              m = (__m128i *)m[10].m128i_i64[0];
            LOBYTE(v14) = v14 | 0x90;
            m = sub_73C240(m);
          }
          else
          {
LABEL_49:
            LOBYTE(v19) = v14 | 0x80;
            v14 = v19;
          }
        }
        else if ( (unsigned int)sub_8D2310(m) )
        {
          for ( k = m; k[8].m128i_i8[12] == 12; k = (__m128i *)k[10].m128i_i64[0] )
            ;
          if ( (unsigned int)sub_8D76D0(k) )
          {
            while ( m[8].m128i_i8[12] == 12 )
              m = (__m128i *)m[10].m128i_i64[0];
            v14 |= 0x11u;
            m = sub_73C240(m);
          }
        }
        if ( !v17 )
          v14 |= 0x20u;
        v15 = sub_724D50(1);
        sub_72BBE0((__int64)v15, v14, unk_4F06870);
        v16 = v24[0];
        if ( *(_QWORD *)(v24[0] + 176LL) )
          *(_QWORD *)(*(_QWORD *)(v24[0] + 184LL) + 120LL) = v15;
        else
          *(_QWORD *)(v24[0] + 176LL) = v15;
        *(_QWORD *)(v16 + 184) = v15;
        ++*(_QWORD *)(*(_QWORD *)(v23 + 120) + 176LL);
      }
      while ( v17 );
      v22 = v16;
      sub_8D6090(*(_QWORD *)(v23 + 120));
      *(_QWORD *)(v22 + 128) = *(_QWORD *)(v23 + 120);
      *a3 = v23;
    }
    else
    {
      for ( m = v8; (unsigned int)sub_8D2E30(m) && !(unsigned int)sub_7E1E50(m); m = (__m128i *)sub_8D46C0(m) )
        ;
    }
    return sub_7E1E20(m);
  }
  *a2 |= 1uLL;
  if ( (v8[8].m128i_i8[12] & 0xFB) == 8 )
  {
    v12 = sub_8D4C10(v8, dword_4F077C4 != 2);
    if ( (v12 & 1) != 0 )
      *a2 |= 2uLL;
    if ( (v12 & 2) != 0 )
      *a2 |= 4uLL;
  }
  if ( (unsigned int)sub_7E1F40(v8) )
  {
    *a2 |= 0x40uLL;
  }
  else if ( (unsigned int)sub_7E1F90(v8) )
  {
    *a2 |= 0x80uLL;
  }
  m = v8;
  if ( !(unsigned int)sub_8D2310(v8) )
    return sub_7E1E20(m);
  for ( n = v8; n[8].m128i_i8[12] == 12; n = (__m128i *)n[10].m128i_i64[0] )
    ;
  m = v8;
  if ( !(unsigned int)sub_8D76D0(n) )
    return sub_7E1E20(m);
  v10 = sub_73C240(v8);
  *a2 |= 0x10uLL;
  return sub_7E1E20(v10);
}
