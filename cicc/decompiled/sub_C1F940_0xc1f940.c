// Function: sub_C1F940
// Address: 0xc1f940
//
void __fastcall sub_C1F940(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 v6; // r8
  unsigned __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r13
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r11
  char v15; // al
  __int64 v16; // r10
  unsigned int v17; // edi
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // r11
  int v21; // eax
  unsigned __int64 v22; // rdx
  bool v23; // zf
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int64 v27; // rdi
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  char v30; // al
  unsigned __int64 v31; // r8
  __m128i *v32; // r13
  __m128i *v33; // rax
  __int8 *v34; // r13
  unsigned __int64 v35; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v36; // [rsp-98h] [rbp-98h]
  __int64 v37; // [rsp-90h] [rbp-90h]
  __int64 v38; // [rsp-80h] [rbp-80h] BYREF
  __int64 v39; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 i; // [rsp-70h] [rbp-70h]
  __m128i v41; // [rsp-68h] [rbp-68h] BYREF
  __m128i v42; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v43; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int64 v44; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v3 = a2 - 2;
    v4 = a2 - 1;
    if ( v3 <= v4 )
      v4 = v3;
    v39 = a1 + 1;
    for ( i = v4; i; ++*(_DWORD *)(a3 + 8) )
    {
      while ( 1 )
      {
        v24 = sub_C931B0(&v39, " @ ", 3, 0);
        if ( v24 == -1 )
        {
          v26 = v39;
          v24 = i;
          v27 = 0;
          v28 = 0;
        }
        else
        {
          v25 = v24 + 3;
          v26 = v39;
          if ( v24 + 3 > i )
          {
            v25 = i;
            v27 = 0;
          }
          else
          {
            v27 = i - v25;
          }
          v28 = v39 + v25;
          if ( v24 > i )
            v24 = i;
        }
        v39 = v28;
        i = v27;
        v41.m128i_i64[0] = v26;
        v41.m128i_i64[1] = v24;
        LOBYTE(v38) = 58;
        v29 = sub_C931B0(&v41, &v38, 1, 0);
        if ( v29 == -1 )
        {
          v8 = v41.m128i_i64[0];
          v17 = 0;
          LODWORD(v16) = 0;
          v42 = _mm_loadu_si128(&v41);
          v9 = v42.m128i_i64[1];
        }
        else
        {
          v6 = v41.m128i_i64[1];
          v7 = v29 + 1;
          v8 = v41.m128i_i64[0];
          if ( v29 + 1 > v41.m128i_i64[1] )
          {
            if ( v29 <= v41.m128i_i64[1] )
              v6 = v29;
            v17 = 0;
            LODWORD(v16) = 0;
            v9 = v6;
          }
          else
          {
            v42.m128i_i64[0] = v41.m128i_i64[0];
            if ( v29 <= v41.m128i_i64[1] )
              v6 = v29;
            v44 = v41.m128i_i64[1] - v7;
            v43 = v41.m128i_i64[0] + v7;
            v42.m128i_i64[1] = v6;
            v9 = v6;
            if ( v41.m128i_i64[1] == v7 )
            {
              v17 = 0;
              LODWORD(v16) = 0;
            }
            else
            {
              LOBYTE(v38) = 46;
              v37 = v41.m128i_i64[0];
              v10 = sub_C931B0(&v43, &v38, 1, 0);
              if ( v10 == -1 )
              {
                v13 = v43;
                v11 = v44;
                v14 = 0;
                v35 = 0;
              }
              else
              {
                v11 = v44;
                v12 = v10 + 1;
                v13 = v43;
                if ( v10 + 1 > v44 )
                {
                  v12 = v44;
                  v14 = 0;
                }
                else
                {
                  v14 = v44 - v12;
                }
                v35 = v43 + v12;
                if ( v10 <= v44 )
                  v11 = v10;
              }
              v36 = v14;
              v15 = sub_C93CC0(v13, v11, 10, &v38);
              v8 = v37;
              if ( v15 || (v16 = v38, v16 != (int)v16) )
                LODWORD(v16) = 0;
              v17 = 0;
              if ( v36 )
              {
                LODWORD(v36) = v16;
                v30 = sub_C93C90(v35, v36, 10, &v38);
                v8 = v37;
                LODWORD(v16) = v36;
                if ( v30 || (v17 = v38, v38 != (unsigned int)v38) )
                  v17 = 0;
              }
            }
          }
        }
        v18 = *(unsigned int *)(a3 + 8);
        v19 = *(_QWORD *)a3;
        v20 = *(unsigned int *)(a3 + 12);
        v21 = *(_DWORD *)(a3 + 8);
        v22 = *(_QWORD *)a3 + 24 * v18;
        if ( v18 >= v20 )
          break;
        if ( v22 )
        {
          *(_QWORD *)v22 = v8;
          *(_QWORD *)(v22 + 8) = v9;
          *(_DWORD *)(v22 + 16) = v16;
          *(_DWORD *)(v22 + 20) = v17;
          v21 = *(_DWORD *)(a3 + 8);
        }
        v23 = i == 0;
        *(_DWORD *)(a3 + 8) = v21 + 1;
        if ( v23 )
          return;
      }
      v31 = v18 + 1;
      v42.m128i_i64[1] = v9;
      v32 = &v42;
      v42.m128i_i64[0] = v8;
      v43 = __PAIR64__(v17, v16);
      if ( v20 < v18 + 1 )
      {
        if ( v19 > (unsigned __int64)&v42 || v22 <= (unsigned __int64)&v42 )
        {
          sub_C8D5F0(a3, a3 + 16, v31, 24);
          v19 = *(_QWORD *)a3;
          v18 = *(unsigned int *)(a3 + 8);
        }
        else
        {
          v34 = &v42.m128i_i8[-v19];
          sub_C8D5F0(a3, a3 + 16, v31, 24);
          v19 = *(_QWORD *)a3;
          v18 = *(unsigned int *)(a3 + 8);
          v32 = (__m128i *)&v34[*(_QWORD *)a3];
        }
      }
      v33 = (__m128i *)(v19 + 24 * v18);
      *v33 = _mm_loadu_si128(v32);
      v33[1].m128i_i64[0] = v32[1].m128i_i64[0];
    }
  }
}
