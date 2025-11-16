// Function: sub_39122E0
// Address: 0x39122e0
//
unsigned __int64 *__fastcall sub_39122E0(unsigned __int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rsi
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rdx
  _DWORD *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned int v19; // edi
  int *v20; // rax
  int v21; // r10d
  unsigned int v22; // edx
  __int32 v23; // ecx
  __m128i *v24; // rsi
  int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rdx
  const __m128i *v28; // r8
  __int32 v29; // ecx
  __int64 v30; // rdi
  int v31; // eax
  __int64 v32; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+0h] [rbp-60h]
  int v34; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __m128i v36; // [rsp+10h] [rbp-50h] BYREF
  __int64 v37; // [rsp+20h] [rbp-40h]

  v4 = a2 + 224;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = v4;
  if ( v7 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v7 + 16);
        v10 = *(_QWORD *)(v7 + 24);
        if ( *(_DWORD *)(v7 + 32) >= a3 )
          break;
        v7 = *(_QWORD *)(v7 + 24);
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v9 );
LABEL_6:
    if ( v4 != v8 && *(_DWORD *)(v8 + 32) <= a3 )
    {
      v35 = v8;
      v12 = sub_390FE40(a2, a3);
      v13 = *(_QWORD *)(v35 + 40);
      if ( v13 != *(_QWORD *)(v35 + 48) )
      {
        v14 = a2;
        v15 = *(_QWORD *)(v35 + 48);
        v16 = v14;
        while ( 1 )
        {
          v27 = *(_QWORD *)(v16 + 264);
          v28 = (const __m128i *)(v27 + 24 * v13);
          v29 = v28->m128i_i32[0];
          if ( v28->m128i_i32[0] == a3 )
            break;
          v17 = (unsigned int)v12[12];
          if ( (_DWORD)v17 )
          {
            v18 = *((_QWORD *)v12 + 4);
            v19 = (v17 - 1) & (37 * v29);
            v20 = (int *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v29 != *v20 )
            {
              v31 = 1;
              while ( v21 != -1 )
              {
                v19 = (v17 - 1) & (v31 + v19);
                v34 = v31 + 1;
                v20 = (int *)(v18 + 16LL * v19);
                v21 = *v20;
                if ( v29 == *v20 )
                  goto LABEL_13;
                v31 = v34;
              }
              goto LABEL_19;
            }
LABEL_13:
            if ( v20 != (int *)(v18 + 16 * v17) )
            {
              v22 = v20[1];
              v23 = v20[2];
              v24 = (__m128i *)a1[1];
              v25 = v20[3];
              if ( v24 != (__m128i *)*a1 )
              {
                if ( v24[-2].m128i_i32[3] == v22 && v24[-1].m128i_i32[0] == v23 && v24[-1].m128i_u16[2] == v25 )
                  goto LABEL_19;
                v26 = v28[1].m128i_i64[0];
                v36.m128i_i8[14] &= 0xFCu;
                v36.m128i_i64[0] = __PAIR64__(v22, a3);
                v36.m128i_i32[2] = v23;
                v36.m128i_i16[6] = v25;
                v37 = v26;
                if ( (__m128i *)a1[2] == v24 )
                {
LABEL_27:
                  v32 = v16;
                  sub_3912130(a1, v24, &v36);
                  v16 = v32;
                  goto LABEL_19;
                }
                goto LABEL_17;
              }
              v30 = v28[1].m128i_i64[0];
              v36.m128i_i8[14] &= 0xFCu;
              v36.m128i_i64[0] = __PAIR64__(v22, a3);
              v36.m128i_i32[2] = v23;
              v36.m128i_i16[6] = v25;
              v37 = v30;
              if ( v24 == (__m128i *)a1[2] )
                goto LABEL_27;
              if ( v24 )
              {
LABEL_17:
                *v24 = _mm_loadu_si128(&v36);
                v24[1].m128i_i64[0] = v37;
                v24 = (__m128i *)a1[1];
              }
LABEL_18:
              a1[1] = (unsigned __int64)&v24[1].m128i_u64[1];
            }
          }
LABEL_19:
          if ( v15 == ++v13 )
            return a1;
        }
        v24 = (__m128i *)a1[1];
        if ( v24 == (__m128i *)a1[2] )
        {
          v33 = v16;
          sub_3911D10(a1, v24, (const __m128i *)(v27 + 24 * v13));
          v16 = v33;
          goto LABEL_19;
        }
        if ( v24 )
        {
          *v24 = _mm_loadu_si128(v28);
          v24[1].m128i_i64[0] = v28[1].m128i_i64[0];
          v24 = (__m128i *)a1[1];
        }
        goto LABEL_18;
      }
    }
  }
  return a1;
}
