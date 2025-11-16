// Function: sub_22A6590
// Address: 0x22a6590
//
char *__fastcall sub_22A6590(__int8 *a1, char *a2, char *a3)
{
  char *v3; // r12
  __int8 *v4; // r11
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  char *v8; // rcx
  __int8 *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r9
  __m128i v13; // xmm0
  int v14; // r14d
  int v15; // r13d
  int v16; // ebx
  __int64 v17; // r10
  __int64 v18; // r9
  int v19; // r15d
  __m128i v20; // xmm2
  __int64 v21; // r14
  __m128i *v22; // rcx
  __m128i *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r9
  __m128i v27; // xmm3
  __int32 v28; // r14d
  __int32 v29; // r13d
  __int32 v30; // ebx
  __int64 v31; // r10
  __int64 v32; // r9
  __int32 v33; // r15d
  __m128i v34; // xmm5
  __int64 v35; // r14
  char *v37; // rdx
  __int8 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rcx
  __m128i v41; // xmm6
  int v42; // r10d
  int v43; // r9d
  int v44; // r8d
  int v45; // ebx
  __int64 v46; // rdi
  __int64 v47; // rcx
  __m128i v48; // xmm6
  __int64 v49; // r10

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 != a3 )
  {
    v3 = &a1[a3 - a2];
    v5 = 0x6DB6DB6DB6DB6DB7LL * ((a3 - a1) >> 3);
    v6 = 0x6DB6DB6DB6DB6DB7LL * ((a2 - a1) >> 3);
    if ( v6 == v5 - v6 )
    {
      v37 = a2;
      v38 = v4;
      do
      {
        v39 = *((_QWORD *)v37 + 6);
        v40 = *((_QWORD *)v38 + 6);
        v38 += 56;
        v37 += 56;
        *((_QWORD *)v38 - 1) = v39;
        v41 = _mm_loadu_si128((const __m128i *)(v37 - 40));
        *((_QWORD *)v37 - 1) = v40;
        v42 = *((_DWORD *)v38 - 9);
        v43 = *((_DWORD *)v38 - 8);
        v44 = *((_DWORD *)v38 - 7);
        v45 = *((_DWORD *)v38 - 10);
        *(__m128i *)(v38 - 40) = v41;
        v46 = *((_QWORD *)v38 - 3);
        v47 = *((_QWORD *)v38 - 2);
        *(__m128i *)(v38 - 24) = _mm_loadu_si128((const __m128i *)(v37 - 24));
        v48 = _mm_loadu_si128((const __m128i *)(v37 - 56));
        *((_DWORD *)v37 - 10) = v45;
        *((_DWORD *)v37 - 9) = v42;
        *((_DWORD *)v37 - 8) = v43;
        *((_DWORD *)v37 - 7) = v44;
        *((_QWORD *)v37 - 3) = v46;
        *((_QWORD *)v37 - 2) = v47;
        v49 = *((_QWORD *)v38 - 7);
        LOBYTE(v43) = *(v38 - 48);
        LOBYTE(v44) = *(v38 - 47);
        LOBYTE(v46) = *(v38 - 46);
        LODWORD(v47) = *((_DWORD *)v38 - 11);
        *(__m128i *)(v38 - 56) = v48;
        *((_QWORD *)v37 - 7) = v49;
        *(v37 - 48) = v43;
        *(v37 - 47) = v44;
        *(v37 - 46) = v46;
        *((_DWORD *)v37 - 11) = v47;
      }
      while ( a2 != v38 );
      return &v4[56 * ((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(a2 - 56 - v4) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 56];
    }
    else
    {
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
      while ( 1 )
      {
        v8 = &v4[56 * v6];
        if ( v7 > 0 )
        {
          v9 = v4;
          v10 = 0;
          do
          {
            v11 = *((_QWORD *)v8 + 6);
            v12 = *((_QWORD *)v9 + 6);
            ++v10;
            v9 += 56;
            v8 += 56;
            *((_QWORD *)v9 - 1) = v11;
            v13 = _mm_loadu_si128((const __m128i *)(v8 - 40));
            *((_QWORD *)v8 - 1) = v12;
            v14 = *((_DWORD *)v9 - 9);
            v15 = *((_DWORD *)v9 - 8);
            v16 = *((_DWORD *)v9 - 7);
            v17 = *((_QWORD *)v9 - 3);
            v18 = *((_QWORD *)v9 - 2);
            v19 = *((_DWORD *)v9 - 10);
            *(__m128i *)(v9 - 40) = v13;
            *(__m128i *)(v9 - 24) = _mm_loadu_si128((const __m128i *)(v8 - 24));
            v20 = _mm_loadu_si128((const __m128i *)(v8 - 56));
            *((_DWORD *)v8 - 10) = v19;
            *((_DWORD *)v8 - 9) = v14;
            *((_DWORD *)v8 - 8) = v15;
            *((_DWORD *)v8 - 7) = v16;
            *((_QWORD *)v8 - 3) = v17;
            *((_QWORD *)v8 - 2) = v18;
            v21 = *((_QWORD *)v9 - 7);
            LOBYTE(v15) = *(v9 - 48);
            LOBYTE(v16) = *(v9 - 47);
            LOBYTE(v17) = *(v9 - 46);
            LODWORD(v18) = *((_DWORD *)v9 - 11);
            *(__m128i *)(v9 - 56) = v20;
            *((_QWORD *)v8 - 7) = v21;
            *(v8 - 48) = v15;
            *(v8 - 47) = v16;
            *(v8 - 46) = v17;
            *((_DWORD *)v8 - 11) = v18;
          }
          while ( v7 != v10 );
          v4 += 56 * v7;
        }
        if ( !(v5 % v6) )
          break;
        v7 = v6;
        v6 -= v5 % v6;
        while ( 1 )
        {
          v5 = v7;
          v7 -= v6;
          if ( v6 < v7 )
            break;
LABEL_12:
          v22 = (__m128i *)&v4[56 * v5];
          v4 = &v22->m128i_i8[-56 * v7];
          if ( v6 > 0 )
          {
            v23 = (__m128i *)((char *)v22 - 56 * v7);
            v24 = 0;
            do
            {
              v25 = v22[-1].m128i_i64[1];
              v26 = v23[-1].m128i_i64[1];
              ++v24;
              v23 = (__m128i *)((char *)v23 - 56);
              v22 = (__m128i *)((char *)v22 - 56);
              v23[3].m128i_i64[0] = v25;
              v27 = _mm_loadu_si128(v22 + 1);
              v22[3].m128i_i64[0] = v26;
              v28 = v23[1].m128i_i32[1];
              v29 = v23[1].m128i_i32[2];
              v30 = v23[1].m128i_i32[3];
              v31 = v23[2].m128i_i64[0];
              v32 = v23[2].m128i_i64[1];
              v33 = v23[1].m128i_i32[0];
              v23[1] = v27;
              v23[2] = _mm_loadu_si128(v22 + 2);
              v34 = _mm_loadu_si128(v22);
              v22[1].m128i_i32[0] = v33;
              v22[1].m128i_i32[1] = v28;
              v22[1].m128i_i32[2] = v29;
              v22[1].m128i_i32[3] = v30;
              v22[2].m128i_i64[0] = v31;
              v22[2].m128i_i64[1] = v32;
              v35 = v23->m128i_i64[0];
              LOBYTE(v29) = v23->m128i_i8[8];
              LOBYTE(v30) = v23->m128i_i8[9];
              LOBYTE(v31) = v23->m128i_i8[10];
              LODWORD(v32) = v23->m128i_i32[3];
              *v23 = v34;
              v22->m128i_i64[0] = v35;
              v22->m128i_i8[8] = v29;
              v22->m128i_i8[9] = v30;
              v22->m128i_i8[10] = v31;
              v22->m128i_i32[3] = v32;
            }
            while ( v6 != v24 );
            v4 -= 56 * v6;
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return v3;
        }
      }
    }
    return v3;
  }
  return a1;
}
