// Function: sub_1012D30
// Address: 0x1012d30
//
__int64 __fastcall sub_1012D30(unsigned __int64 a1, _BYTE *a2, __int64 a3, const __m128i *a4, int a5)
{
  __int64 v5; // r14
  char v8; // al
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rax
  int v16; // edx
  unsigned __int64 v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __m128i v20; // xmm3
  __int64 v21; // rdx
  unsigned int v24; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v25; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v26; // [rsp+10h] [rbp-A0h]
  unsigned int v27; // [rsp+1Ch] [rbp-94h]
  __int64 v28; // [rsp+20h] [rbp-90h]
  _BYTE *v29; // [rsp+28h] [rbp-88h]
  __m128i v30[2]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v31; // [rsp+50h] [rbp-60h]
  __m128i v32; // [rsp+60h] [rbp-50h]
  __int64 v33; // [rsp+70h] [rbp-40h]

  v29 = a2;
  v24 = a1;
  if ( a5 )
  {
    v5 = a3;
    if ( *a2 == 84 )
    {
      v5 = (__int64)a2;
      v29 = (_BYTE *)a3;
      v25 = BYTE4(a1);
      v8 = sub_FFE760(a3, (__int64)a2, a4[1].m128i_i64[1]);
    }
    else
    {
      v25 = 0;
      v24 = sub_B52F50(a1);
      v8 = sub_FFE760((__int64)a2, v5, a4[1].m128i_i64[1]);
    }
    if ( v8 && (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) != 0 )
    {
      v9 = 0;
      v27 = a5 - 1;
      v28 = 8LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
      v10 = 0;
      v26 = (unsigned __int64)v25 << 32;
      while ( 1 )
      {
        v12 = *(_QWORD *)(v5 - 8);
        v13 = *(_BYTE **)(v12 + 4 * v10);
        v14 = (_QWORD *)(*(_QWORD *)(32LL * *(unsigned int *)(v5 + 72) + v12 + v10) + 48LL);
        v15 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v13 )
        {
          if ( v14 == (_QWORD *)v15 )
          {
            v17 = 0;
            goto LABEL_17;
          }
        }
        else if ( v14 == (_QWORD *)v15 )
        {
          v17 = 0;
LABEL_18:
          v18 = _mm_loadu_si128(a4);
          v19 = _mm_loadu_si128(a4 + 1);
          v20 = _mm_loadu_si128(a4 + 3);
          v31 = _mm_loadu_si128(a4 + 2);
          v21 = a4[4].m128i_i64[0];
          v31.m128i_i64[1] = v17;
          v33 = v21;
          v30[0] = v18;
          a1 = v26 | v24 | a1 & 0xFFFFFF0000000000LL;
          v30[1] = v19;
          v32 = v20;
          if ( v24 - 32 <= 9 )
          {
            v11 = sub_1012FB0(a1, v13, v29, v30);
            if ( !v11 )
              return 0;
          }
          else
          {
            v11 = sub_1011B90(a1, v13, v29, 0, v30, v27);
            if ( !v11 )
              return 0;
          }
          if ( v9 && v11 != v9 )
            return 0;
          v9 = v11;
          goto LABEL_11;
        }
        if ( !v15 )
          BUG();
        v16 = *(unsigned __int8 *)(v15 - 24);
        v17 = v15 - 24;
        if ( (unsigned int)(v16 - 30) >= 0xB )
          v17 = 0;
LABEL_17:
        if ( (_BYTE *)v5 != v13 )
          goto LABEL_18;
LABEL_11:
        v10 += 8;
        if ( v28 == v10 )
          return v9;
      }
    }
  }
  return 0;
}
