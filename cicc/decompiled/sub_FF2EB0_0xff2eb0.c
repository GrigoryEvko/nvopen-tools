// Function: sub_FF2EB0
// Address: 0xff2eb0
//
char __fastcall sub_FF2EB0(__int64 a1, __m128i **a2, __int64 a3, __int64 a4, int a5, __int64 a6, __int64 a7)
{
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // r13
  __int64 *v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int v14; // eax
  __m128i *v15; // r12
  __int64 v16; // r9
  const __m128i *v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // r8
  unsigned int v21; // eax
  const void *v22; // rsi
  char *v23; // r12
  __int64 v28[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29[10]; // [rsp+30h] [rbp-50h] BYREF

  v8 = *a2;
  if ( *a2 )
  {
    v9 = (unsigned int)(v8[2].m128i_i32[3] + 1);
    LODWORD(v8) = v8[2].m128i_i32[3] + 1;
  }
  else
  {
    v9 = 0;
    LODWORD(v8) = 0;
  }
  if ( (unsigned int)v8 < *(_DWORD *)(a3 + 32) )
  {
    v10 = 0;
    v11 = *(__int64 **)(*(_QWORD *)(a3 + 24) + 8 * v9);
    if ( (unsigned int)v8 < *(_DWORD *)(a4 + 56) )
    {
      v8 = *(__m128i **)(a4 + 48);
      v10 = (__m128i *)v8->m128i_i64[v9];
    }
    for ( ; v11; v11 = (__int64 *)v11[1] )
    {
      v12 = *v11;
      if ( *v11 )
      {
        v13 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
        v14 = *(_DWORD *)(v12 + 44) + 1;
      }
      else
      {
        v13 = 0;
        v14 = 0;
      }
      if ( v14 < *(_DWORD *)(a4 + 56) )
      {
        v8 = *(__m128i **)(a4 + 48);
        v15 = (__m128i *)v8->m128i_i64[v13];
        if ( v15 )
        {
          if ( v10 != v15 )
          {
            if ( !v10 )
              return (char)v8;
            if ( v10 != (__m128i *)v15->m128i_i64[1] )
            {
              if ( v15 == (__m128i *)v10->m128i_i64[1] )
                return (char)v8;
              LODWORD(v8) = v15[1].m128i_i32[0];
              if ( v10[1].m128i_i32[0] >= (unsigned int)v8 )
                return (char)v8;
              if ( *(_BYTE *)(a4 + 136) )
              {
                LODWORD(v8) = v10[4].m128i_i32[2];
                if ( v15[4].m128i_i32[2] < (unsigned int)v8 )
                  return (char)v8;
                LODWORD(v8) = v10[4].m128i_i32[3];
                if ( v15[4].m128i_i32[3] > (unsigned int)v8 )
                  return (char)v8;
              }
              else
              {
                v21 = *(_DWORD *)(a4 + 140) + 1;
                *(_DWORD *)(a4 + 140) = v21;
                if ( v21 > 0x20 )
                {
                  sub_B19820(a4);
                  LODWORD(v8) = v10[4].m128i_i32[2];
                  if ( v15[4].m128i_i32[2] < (unsigned int)v8 )
                    return (char)v8;
                  LODWORD(v8) = v10[4].m128i_i32[3];
                  if ( v15[4].m128i_i32[3] > (unsigned int)v8 )
                    return (char)v8;
                }
                else
                {
                  do
                  {
                    v8 = v15;
                    v15 = (__m128i *)v15->m128i_i64[1];
                  }
                  while ( v15 && v10[1].m128i_i32[0] <= (unsigned __int32)v15[1].m128i_i32[0] );
                  if ( v10 != v8 )
                    return (char)v8;
                }
              }
            }
          }
        }
      }
      sub_FEF2D0((__int64)v29, v12, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
      v28[0] = (__int64)v29;
      v28[1] = (__int64)a2;
      if ( sub_FEF400(a1, v28) )
      {
        LOBYTE(v8) = sub_FEF3D0(a1, v28);
        if ( (_BYTE)v8 )
        {
          v17 = (const __m128i *)v29;
          v18 = *(unsigned int *)(a7 + 8);
          v19 = *(_QWORD *)a7;
          v20 = v18 + 1;
          if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a7 + 12) )
          {
            v22 = (const void *)(a7 + 16);
            if ( v19 > (unsigned __int64)v29 || (unsigned __int64)v29 >= v19 + 24 * v18 )
            {
              sub_C8D5F0(a7, v22, v20, 0x18u, v20, v16);
              v17 = (const __m128i *)v29;
              v19 = *(_QWORD *)a7;
              v18 = *(unsigned int *)(a7 + 8);
            }
            else
            {
              v23 = (char *)v29 - v19;
              sub_C8D5F0(a7, v22, v20, 0x18u, v20, v16);
              v19 = *(_QWORD *)a7;
              v18 = *(unsigned int *)(a7 + 8);
              v17 = (const __m128i *)&v23[*(_QWORD *)a7];
            }
          }
          v8 = (__m128i *)(v19 + 24 * v18);
          *v8 = _mm_loadu_si128(v17);
          v8[1].m128i_i64[0] = v17[1].m128i_i64[0];
          LOBYTE(v8) = a7;
          ++*(_DWORD *)(a7 + 8);
        }
      }
      else
      {
        LOBYTE(v8) = sub_FF2910(a1, v29, a5, a6, a7);
        if ( !(_BYTE)v8 )
          return (char)v8;
      }
    }
  }
  return (char)v8;
}
