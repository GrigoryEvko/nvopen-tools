// Function: sub_A7AD50
// Address: 0xa7ad50
//
__int64 __fastcall sub_A7AD50(_QWORD *a1, unsigned int *a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  unsigned int v5; // r15d
  unsigned int v6; // eax
  unsigned int v7; // r15d
  unsigned __int64 *v8; // rax
  __int64 v9; // rax
  unsigned int *v10; // rsi
  __m128i v11; // rax
  __int64 v12; // r10
  __int64 v13; // rax
  unsigned __int64 v14; // r9
  unsigned __int64 v15; // rdx
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rbx
  unsigned int *v20; // r12
  unsigned __int64 v21; // rax
  unsigned int *v22; // r11
  unsigned int v23; // ecx
  unsigned int v24; // edx
  unsigned int *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned int *v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+10h] [rbp-D0h]
  unsigned int v32; // [rsp+24h] [rbp-BCh]
  __int64 v33; // [rsp+48h] [rbp-98h] BYREF
  __m128i v34; // [rsp+50h] [rbp-90h] BYREF
  __m128i v35; // [rsp+60h] [rbp-80h] BYREF
  __m128i v36; // [rsp+70h] [rbp-70h] BYREF
  _BYTE v37[96]; // [rsp+80h] [rbp-60h] BYREF

  v33 = a3;
  if ( a3 == *a1 )
  {
    v36.m128i_i64[0] = a3;
    v36.m128i_i8[8] = 1;
    v35 = _mm_load_si128(&v36);
  }
  else
  {
    v36.m128i_i64[0] = (__int64)v37;
    v36.m128i_i64[1] = 0x300000000LL;
    v5 = sub_A74480((__int64)&v33);
    v6 = sub_A74480((__int64)a1);
    if ( v5 >= v6 )
      v6 = v5;
    v7 = -1;
    v32 = v6 - 1;
    if ( v6 )
    {
      while ( 1 )
      {
        v35.m128i_i64[0] = sub_A74490(a1, v7);
        v9 = sub_A74490(&v33, v7);
        v10 = a2;
        v11.m128i_i64[0] = sub_A7A6D0(v35.m128i_i64, (__int64)a2, v9);
        v34 = v11;
        v12 = v11.m128i_i64[0];
        if ( !v11.m128i_i8[8] )
          break;
        if ( v11.m128i_i64[0] )
        {
          v13 = v36.m128i_u32[2];
          v14 = v3 & 0xFFFFFFFF00000000LL | v7;
          v15 = v36.m128i_u32[2] + 1LL;
          v3 = v14;
          if ( v15 > v36.m128i_u32[3] )
          {
            v30 = v12;
            v31 = v14;
            sub_C8D5F0(&v36, v37, v15, 16);
            v13 = v36.m128i_u32[2];
            v12 = v30;
            v14 = v31;
          }
          v8 = (unsigned __int64 *)(v36.m128i_i64[0] + 16 * v13);
          *v8 = v14;
          v8[1] = v12;
          ++v36.m128i_i32[2];
        }
        if ( v32 == ++v7 )
          goto LABEL_14;
      }
      v35.m128i_i8[8] = 0;
    }
    else
    {
LABEL_14:
      v17 = v36.m128i_u32[2];
      v18 = v36.m128i_i64[0];
      v19 = 16LL * v36.m128i_u32[2];
      v20 = (unsigned int *)(v36.m128i_i64[0] + v19);
      if ( v36.m128i_i64[0] != v36.m128i_i64[0] + v19 )
      {
        _BitScanReverse64(&v21, v19 >> 4);
        sub_A790C0(v36.m128i_i64[0], v36.m128i_i64[0] + v19, 2LL * (int)(63 - (v21 ^ 0x3F)));
        if ( (unsigned __int64)v19 <= 0x100 )
        {
          sub_A6E0B0(v18, v20);
        }
        else
        {
          sub_A6E0B0(v18, (unsigned int *)(v18 + 256));
          for ( ; v20 != v22; *((_QWORD *)v28 + 1) = v26 )
          {
            v23 = *v22;
            v24 = *(v22 - 4);
            v25 = v22 - 4;
            v26 = *((_QWORD *)v22 + 1);
            if ( *v22 >= v24 )
            {
              v28 = v22;
            }
            else
            {
              do
              {
                v25[4] = v24;
                v27 = *((_QWORD *)v25 + 1);
                v28 = v25;
                v25 -= 4;
                *((_QWORD *)v25 + 5) = v27;
                v24 = *v25;
              }
              while ( v23 < *v25 );
            }
            v22 += 4;
            *v28 = v23;
          }
        }
        v20 = (unsigned int *)v36.m128i_i64[0];
        v17 = v36.m128i_u32[2];
      }
      v10 = v20;
      v29 = sub_A78010(a2, (int *)v20, v17);
      v34.m128i_i8[8] = 1;
      v34.m128i_i64[0] = v29;
      v35 = _mm_load_si128(&v34);
    }
    if ( (_BYTE *)v36.m128i_i64[0] != v37 )
      _libc_free(v36.m128i_i64[0], v10);
  }
  return v35.m128i_i64[0];
}
