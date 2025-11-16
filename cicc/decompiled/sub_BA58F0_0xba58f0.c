// Function: sub_BA58F0
// Address: 0xba58f0
//
void __fastcall sub_BA58F0(__int64 a1)
{
  __int64 v2; // rax
  int v3; // ecx
  _BYTE *v4; // rsi
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // edx
  const __m128i *v10; // rbx
  const __m128i *v11; // r14
  __int64 *v12; // rdi
  const __m128i *v13; // rax
  __int64 v14; // r15
  const __m128i *v15; // rdx
  __m128i *v16; // rcx
  const __m128i *v17; // rax
  __int64 *v18; // r14
  __int64 *v19; // rbx
  __int8 v20; // al
  __int64 v21; // rax
  __m128i *v22; // r15
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int8 *v26; // rax
  int v27; // eax
  int v28; // r9d
  __int64 *v29; // [rsp-108h] [rbp-108h] BYREF
  __int64 v30; // [rsp-100h] [rbp-100h]
  _BYTE v31[248]; // [rsp-F8h] [rbp-F8h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 8) != 0 )
  {
    v2 = ***(_QWORD ***)(a1 + 8);
    v3 = *(_DWORD *)(v2 + 592);
    v4 = *(_BYTE **)(v2 + 576);
    if ( v3 )
    {
      v5 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v6 = (__int64 *)&v4[16 * v5];
      v7 = *v6;
      if ( a1 == *v6 )
      {
LABEL_4:
        v8 = v6[1];
        v9 = *(_DWORD *)(v8 + 32) >> 1;
        if ( (*(_BYTE *)(v8 + 32) & 1) != 0 )
        {
          v10 = (const __m128i *)(v8 + 136);
          v11 = (const __m128i *)(v8 + 40);
          if ( v9 )
            goto LABEL_13;
          goto LABEL_6;
        }
        v11 = *(const __m128i **)(v8 + 40);
        v10 = (const __m128i *)((char *)v11 + 24 * *(unsigned int *)(v8 + 48));
        if ( !v9 )
        {
LABEL_6:
          v12 = (__int64 *)v31;
LABEL_7:
          if ( v12 != (__int64 *)v31 )
            _libc_free(v12, v4);
          return;
        }
        while ( 1 )
        {
LABEL_13:
          if ( v11 == v10 )
            goto LABEL_6;
          if ( v11->m128i_i64[0] != -4096 && v11->m128i_i64[0] != -8192 )
            break;
          v11 = (const __m128i *)((char *)v11 + 24);
        }
        v29 = (__int64 *)v31;
        v30 = 0x800000000LL;
        if ( v11 == v10 )
          goto LABEL_6;
        v13 = v11;
        v14 = 0;
        while ( 1 )
        {
          v15 = (const __m128i *)((char *)v13 + 24);
          if ( &v13[1].m128i_u64[1] == (unsigned __int64 *)v10 )
            break;
          while ( 1 )
          {
            v13 = v15;
            if ( v15->m128i_i64[0] != -8192 && v15->m128i_i64[0] != -4096 )
              break;
            v15 = (const __m128i *)((char *)v15 + 24);
            if ( v10 == v15 )
              goto LABEL_21;
          }
          ++v14;
          if ( v15 == v10 )
            goto LABEL_22;
        }
LABEL_21:
        ++v14;
LABEL_22:
        v16 = (__m128i *)v31;
        if ( v14 > 8 )
        {
          v4 = v31;
          sub_C8D5F0(&v29, v31, v14, 24);
          v16 = (__m128i *)&v29[3 * (unsigned int)v30];
        }
        do
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(v11);
            v16[1].m128i_i64[0] = v11[1].m128i_i64[0];
          }
          v17 = (const __m128i *)((char *)v11 + 24);
          if ( &v11[1].m128i_u64[1] == (unsigned __int64 *)v10 )
            break;
          while ( 1 )
          {
            v11 = v17;
            if ( v17->m128i_i64[0] != -8192 && v17->m128i_i64[0] != -4096 )
              break;
            v17 = (const __m128i *)((char *)v17 + 24);
            if ( v10 == v17 )
              goto LABEL_30;
          }
          v16 = (__m128i *)((char *)v16 + 24);
        }
        while ( v17 != v10 );
LABEL_30:
        v12 = v29;
        LODWORD(v30) = v30 + v14;
        v18 = &v29[3 * (unsigned int)v30];
        if ( v18 == v29 )
          goto LABEL_7;
        v19 = v29;
        while ( 1 )
        {
          v21 = v19[1];
          v22 = (__m128i *)(v21 & 0xFFFFFFFFFFFFFFFCLL);
          if ( (v21 & 0xFFFFFFFFFFFFFFFCLL) == 0 )
            goto LABEL_36;
          v23 = v21 & 3;
          if ( v23 )
          {
            if ( v23 != 1 )
              goto LABEL_36;
            v20 = v22->m128i_i8[0];
            if ( (unsigned __int8)(v22->m128i_i8[0] - 5) > 0x1Fu )
              goto LABEL_36;
            if ( (unsigned __int8)v20 > 0x1Eu )
            {
              if ( (unsigned __int8)(v20 - 33) <= 3u )
              {
LABEL_46:
                v25 = sub_ACADE0(*(__int64 ***)(a1 + 8));
                v26 = (unsigned __int8 *)sub_B98A20(v25, (__int64)v4);
                v4 = (_BYTE *)*v19;
                sub_BA56C0(v22, *v19, v26);
              }
            }
            else if ( (unsigned __int8)v20 > 8u )
            {
              goto LABEL_46;
            }
LABEL_36:
            v19 += 3;
            if ( v18 == v19 )
              goto LABEL_40;
          }
          else
          {
            v19 += 3;
            v24 = sub_ACADE0(*(__int64 ***)(a1 + 8));
            v4 = sub_B98A20(v24, (__int64)v4);
            sub_B9F930((__int64)v22, v4);
            if ( v18 == v19 )
            {
LABEL_40:
              v12 = v29;
              goto LABEL_7;
            }
          }
        }
      }
      v27 = 1;
      while ( v7 != -4096 )
      {
        v28 = v27 + 1;
        v5 = (v3 - 1) & (v27 + v5);
        v6 = (__int64 *)&v4[16 * v5];
        v7 = *v6;
        if ( a1 == *v6 )
          goto LABEL_4;
        v27 = v28;
      }
    }
    v6 = (__int64 *)&v4[16 * v3];
    goto LABEL_4;
  }
}
