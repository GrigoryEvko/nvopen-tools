// Function: sub_264A8F0
// Address: 0x264a8f0
//
_BOOL8 __fastcall sub_264A8F0(__int64 a1, __m128i *a2, unsigned int a3, unsigned __int64 *a4, _BYTE *a5)
{
  __int64 v6; // r15
  __int64 v7; // r12
  char v8; // al
  unsigned __int8 v9; // cl
  bool v10; // bl
  __int64 v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+18h] [rbp-58h]
  bool v17; // [rsp+26h] [rbp-4Ah]
  bool v18; // [rsp+27h] [rbp-49h]
  char v19; // [rsp+27h] [rbp-49h]
  __m128i v21; // [rsp+30h] [rbp-40h] BYREF

  v15 = (__int64)a2;
  if ( (unsigned int)qword_4FF37E8 < a3 )
    return 0;
  if ( a2->m128i_i8[0] )
  {
    if ( a2->m128i_i8[0] != 1 )
      BUG();
    v15 = a2[-2].m128i_i64[0];
    if ( *(_BYTE *)v15 )
      BUG();
  }
  v13 = *(_QWORD *)(v15 + 80);
  if ( v13 == v15 + 72 )
    return 0;
  v18 = 0;
  do
  {
    if ( !v13 )
      BUG();
    v6 = *(_QWORD *)(v13 + 32);
    v7 = v13 + 24;
    if ( v6 != v13 + 24 )
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        v9 = *(_BYTE *)(v6 - 24) - 34;
        if ( v9 > 0x33u )
          goto LABEL_15;
        v17 = ((0x8000000000041uLL >> v9) & 1) == 0;
        if ( (~(0x8000000000041uLL >> v9) & 1) != 0 )
          goto LABEL_15;
        v10 = sub_B49220(v6 - 24);
        if ( !v10 )
          goto LABEL_15;
        v11 = *(_QWORD *)(v6 - 56);
        if ( !v11 )
          BUG();
        if ( *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(v6 + 56) )
        {
          v11 = (__int64)sub_BD3990((unsigned __int8 *)v11, (__int64)a2);
          if ( *(_BYTE *)v11 )
          {
            if ( *(_BYTE *)v11 != 1 )
              goto LABEL_15;
            v11 = sub_B325F0(v11);
            if ( *(_BYTE *)v11 )
              goto LABEL_15;
          }
        }
        if ( a1 == v11 )
        {
          if ( v18 )
          {
LABEL_42:
            *a5 = 1;
            return v17;
          }
          a2 = (__m128i *)a4[1];
          v21.m128i_i64[0] = v6 - 24;
          v21.m128i_i64[1] = v15;
          if ( a2 == (__m128i *)a4[2] )
          {
            sub_264A770(a4, a2, &v21);
          }
          else
          {
            if ( a2 )
            {
              *a2 = _mm_loadu_si128(&v21);
              a2 = (__m128i *)a4[1];
            }
            a4[1] = (unsigned __int64)++a2;
          }
          v6 = *(_QWORD *)(v6 + 8);
          v18 = v10;
          if ( v7 == v6 )
            break;
        }
        else
        {
          a2 = (__m128i *)v11;
          v8 = sub_264A8F0(a1, v11, a3 + 1, a4, a5);
          if ( v8 )
          {
            if ( v18 )
              goto LABEL_42;
            a2 = (__m128i *)a4[1];
            v21.m128i_i64[0] = v6 - 24;
            v21.m128i_i64[1] = v15;
            if ( a2 == (__m128i *)a4[2] )
            {
              v19 = v8;
              sub_264A770(a4, a2, &v21);
              v8 = v19;
            }
            else
            {
              if ( a2 )
              {
                *a2 = _mm_loadu_si128(&v21);
                a2 = (__m128i *)a4[1];
              }
              a4[1] = (unsigned __int64)++a2;
            }
            v18 = v8;
          }
          else if ( *a5 )
          {
            return 0;
          }
LABEL_15:
          v6 = *(_QWORD *)(v6 + 8);
          if ( v7 == v6 )
            break;
        }
      }
    }
    v13 = *(_QWORD *)(v13 + 8);
  }
  while ( v15 + 72 != v13 );
  return v18;
}
