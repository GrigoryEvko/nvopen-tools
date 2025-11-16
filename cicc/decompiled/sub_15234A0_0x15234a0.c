// Function: sub_15234A0
// Address: 0x15234a0
//
char __fastcall sub_15234A0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r14
  char *v6; // rdx
  unsigned __int64 v7; // rax
  char *v8; // r15
  __int64 v9; // r14
  __m128i *v10; // rsi
  unsigned __int64 v11; // rsi
  char *v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  char *v18; // [rsp+8h] [rbp-48h]
  __m128i v19; // [rsp+10h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(char **)a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v5 - *(_QWORD *)a1) >> 3);
  if ( (_DWORD)v7 == a3 )
  {
    v19.m128i_i64[0] = a2;
    if ( v5 == *(_QWORD *)(a1 + 16) )
    {
      LOBYTE(v7) = sub_14EFA10((char **)a1, (char *)v5, &v19);
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = 6;
        *(_QWORD *)(v5 + 8) = 0;
        LOBYTE(v7) = a2 != -8;
        *(_QWORD *)(v5 + 16) = a2;
        if ( ((unsigned __int8)v7 & (a2 != 0)) != 0 && a2 != -16 )
          LOBYTE(v7) = sub_164C220(v5);
        v5 = *(_QWORD *)(a1 + 8);
      }
      *(_QWORD *)(a1 + 8) = v5 + 24;
    }
  }
  else
  {
    if ( (unsigned int)v7 <= a3 )
    {
      v11 = a3 + 1;
      if ( v11 > v7 )
      {
        sub_14EF7A0(a1, v11 - v7);
        v6 = *(char **)a1;
      }
      else if ( v11 < v7 )
      {
        v18 = &v6[24 * v11];
        if ( (char *)v5 != v18 )
        {
          v12 = &v6[24 * v11];
          do
          {
            v13 = *((_QWORD *)v12 + 2);
            if ( v13 != -8 && v13 != 0 && v13 != -16 )
              sub_1649B30(v12);
            v12 += 24;
          }
          while ( (char *)v5 != v12 );
          v6 = *(char **)a1;
          *(_QWORD *)(a1 + 8) = v18;
        }
      }
    }
    LOBYTE(v7) = 3 * a3;
    v8 = &v6[24 * a3];
    v9 = *((_QWORD *)v8 + 2);
    if ( v9 )
    {
      if ( *(_BYTE *)(v9 + 16) > 0x10u )
      {
        sub_164D160(*((_QWORD *)v8 + 2), a2);
        LOBYTE(v7) = sub_164BEC0(v9, a2, v14, v15, v16);
        return v7;
      }
      v19.m128i_i64[0] = *((_QWORD *)v8 + 2);
      v10 = *(__m128i **)(a1 + 32);
      v19.m128i_i32[2] = a3;
      if ( v10 == *(__m128i **)(a1 + 40) )
      {
        sub_1523320((const __m128i **)(a1 + 24), v10, &v19);
      }
      else
      {
        if ( v10 )
        {
          *v10 = _mm_loadu_si128(&v19);
          v10 = *(__m128i **)(a1 + 32);
        }
        *(_QWORD *)(a1 + 32) = v10 + 1;
      }
      v7 = *((_QWORD *)v8 + 2);
      if ( a2 != v7 )
      {
        if ( v7 != -8 && v7 != 0 && v7 != -16 )
          sub_1649B30(v8);
        *((_QWORD *)v8 + 2) = a2;
        LOBYTE(v7) = a2 != -8;
        if ( ((unsigned __int8)v7 & (a2 != 0)) != 0 && a2 != -16 )
          goto LABEL_15;
      }
    }
    else if ( a2 )
    {
      *((_QWORD *)v8 + 2) = a2;
      if ( a2 != -16 && a2 != -8 )
LABEL_15:
        LOBYTE(v7) = sub_164C220(v8);
    }
  }
  return v7;
}
