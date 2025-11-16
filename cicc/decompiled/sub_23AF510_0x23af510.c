// Function: sub_23AF510
// Address: 0x23af510
//
__int64 __fastcall sub_23AF510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __m128i v5; // xmm2
  __int64 v6; // rdi
  _BYTE *v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r12
  __int64 v10; // r14
  const __m128i *v11; // r12
  const __m128i *v12; // r13
  __m128i v13; // xmm1
  unsigned __int64 v14; // r12
  char *v15; // rcx
  __int64 v16; // r12
  void *v17; // rcx
  __m128i v18; // xmm3
  size_t v19; // r12
  _QWORD *v20; // r12
  _QWORD *v21; // rax
  _QWORD *v22; // rcx
  unsigned __int64 v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  __m128i v27; // xmm4
  size_t v28; // r12
  void *v29; // rax
  void *v30; // rcx
  __int64 *v31; // r12
  _QWORD *v32; // rax
  _QWORD *v33; // rcx
  unsigned __int64 v34; // rax
  __m128i v35; // xmm5
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // r13
  __int64 v39; // rax
  __int64 **v40; // rax

  if ( a1 != a2 )
  {
    v4 = a1;
    do
    {
      if ( a3 )
      {
        v5 = _mm_loadu_si128((const __m128i *)v4);
        v6 = a3 + 16;
        *(_QWORD *)(a3 + 16) = a3 + 32;
        *(__m128i *)a3 = v5;
        v7 = *(_BYTE **)(v4 + 16);
        sub_23AEDD0((__int64 *)(a3 + 16), v7, (__int64)&v7[*(_QWORD *)(v4 + 24)]);
        v9 = *(_QWORD *)(v4 + 56) - *(_QWORD *)(v4 + 48);
        *(_QWORD *)(a3 + 48) = 0;
        *(_QWORD *)(a3 + 56) = 0;
        *(_QWORD *)(a3 + 64) = 0;
        if ( v9 )
        {
          if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_47;
          v6 = v9;
          v10 = sub_22077B0(v9);
        }
        else
        {
          v9 = 0;
          v10 = 0;
        }
        *(_QWORD *)(a3 + 48) = v10;
        *(_QWORD *)(a3 + 56) = v10;
        *(_QWORD *)(a3 + 64) = v10 + v9;
        v11 = *(const __m128i **)(v4 + 56);
        if ( v11 != *(const __m128i **)(v4 + 48) )
        {
          v12 = *(const __m128i **)(v4 + 48);
          do
          {
            if ( v10 )
            {
              v13 = _mm_loadu_si128(v12);
              v6 = v10 + 16;
              *(_QWORD *)(v10 + 16) = v10 + 32;
              *(__m128i *)v10 = v13;
              v7 = (_BYTE *)v12[1].m128i_i64[0];
              sub_23AEDD0((__int64 *)(v10 + 16), v7, (__int64)&v7[v12[1].m128i_i64[1]]);
              v8 = v12[3].m128i_u64[0];
              *(_QWORD *)(v10 + 48) = v8;
            }
            v12 = (const __m128i *)((char *)v12 + 56);
            v10 += 56;
          }
          while ( v11 != v12 );
        }
        *(_QWORD *)(a3 + 56) = v10;
        v14 = *(_QWORD *)(v4 + 80) - *(_QWORD *)(v4 + 72);
        *(_QWORD *)(a3 + 72) = 0;
        *(_QWORD *)(a3 + 80) = 0;
        *(_QWORD *)(a3 + 88) = 0;
        if ( v14 )
        {
          if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_47;
          v6 = v14;
          v15 = (char *)sub_22077B0(v14);
        }
        else
        {
          v14 = 0;
          v15 = 0;
        }
        *(_QWORD *)(a3 + 72) = v15;
        *(_QWORD *)(a3 + 88) = &v15[v14];
        *(_QWORD *)(a3 + 80) = v15;
        v7 = *(_BYTE **)(v4 + 72);
        v16 = *(_QWORD *)(v4 + 80) - (_QWORD)v7;
        if ( *(_BYTE **)(v4 + 80) != v7 )
        {
          v6 = (__int64)v15;
          v15 = (char *)memmove(v15, v7, *(_QWORD *)(v4 + 80) - (_QWORD)v7);
        }
        *(_QWORD *)(a3 + 96) = 0;
        *(_QWORD *)(a3 + 80) = &v15[v16];
        v8 = *(_QWORD *)(v4 + 104);
        v17 = (void *)(a3 + 144);
        *(_QWORD *)(a3 + 112) = 0;
        *(_QWORD *)(a3 + 104) = v8;
        *(_QWORD *)(a3 + 120) = *(_QWORD *)(v4 + 120);
        v18 = _mm_loadu_si128((const __m128i *)(v4 + 128));
        *(_QWORD *)(a3 + 144) = 0;
        *(__m128i *)(a3 + 128) = v18;
        if ( v8 != 1 )
        {
          if ( v8 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_47;
          v19 = 8 * v8;
          v7 = 0;
          v6 = sub_22077B0(8 * v8);
          v17 = memset((void *)v6, 0, v19);
        }
        *(_QWORD *)(a3 + 96) = v17;
        v20 = *(_QWORD **)(v4 + 112);
        if ( v20 )
        {
          v6 = 16;
          v21 = (_QWORD *)sub_22077B0(0x10u);
          v22 = v21;
          if ( v21 )
            *v21 = 0;
          v23 = v20[1];
          v7 = (_BYTE *)(a3 + 112);
          *(_QWORD *)(a3 + 112) = v22;
          v22[1] = v23;
          *(_QWORD *)(*(_QWORD *)(a3 + 96) + 8 * (v23 % *(_QWORD *)(a3 + 104))) = a3 + 112;
          while ( 1 )
          {
            v20 = (_QWORD *)*v20;
            if ( !v20 )
              break;
            while ( 1 )
            {
              v6 = 16;
              v24 = v22;
              v25 = (_QWORD *)sub_22077B0(0x10u);
              v22 = v25;
              if ( v25 )
                *v25 = 0;
              v25[1] = v20[1];
              *v24 = v25;
              v26 = (_QWORD *)(*(_QWORD *)(a3 + 96) + 8LL * (v25[1] % *(_QWORD *)(a3 + 104)));
              if ( *v26 )
                break;
              *v26 = v24;
              v20 = (_QWORD *)*v20;
              if ( !v20 )
                goto LABEL_29;
            }
          }
        }
LABEL_29:
        *(_QWORD *)(a3 + 152) = 0;
        v8 = *(_QWORD *)(v4 + 160);
        *(_QWORD *)(a3 + 168) = 0;
        *(_QWORD *)(a3 + 160) = v8;
        *(_QWORD *)(a3 + 176) = *(_QWORD *)(v4 + 176);
        v27 = _mm_loadu_si128((const __m128i *)(v4 + 184));
        *(_QWORD *)(a3 + 200) = 0;
        *(__m128i *)(a3 + 184) = v27;
        if ( v8 == 1 )
        {
          v30 = (void *)(a3 + 200);
        }
        else
        {
          if ( v8 > 0xFFFFFFFFFFFFFFFLL )
LABEL_47:
            sub_4261EA(v6, v7, v8);
          v28 = 8 * v8;
          v29 = (void *)sub_22077B0(8 * v8);
          v30 = memset(v29, 0, v28);
        }
        *(_QWORD *)(a3 + 152) = v30;
        v31 = *(__int64 **)(v4 + 168);
        if ( v31 )
        {
          v32 = (_QWORD *)sub_22077B0(0x18u);
          v33 = v32;
          if ( v32 )
            *v32 = 0;
          v34 = v31[1];
          v35 = _mm_loadu_si128((const __m128i *)(v31 + 1));
          *(_QWORD *)(a3 + 168) = v33;
          v36 = v34 % *(_QWORD *)(a3 + 160);
          v37 = *(_QWORD *)(a3 + 152);
          *(__m128i *)(v33 + 1) = v35;
          *(_QWORD *)(v37 + 8 * v36) = a3 + 168;
          while ( 1 )
          {
            v31 = (__int64 *)*v31;
            if ( !v31 )
              break;
            while ( 1 )
            {
              v38 = v33;
              v39 = sub_22077B0(0x18u);
              v33 = (_QWORD *)v39;
              if ( v39 )
                *(_QWORD *)v39 = 0;
              *(__m128i *)(v39 + 8) = _mm_loadu_si128((const __m128i *)(v31 + 1));
              *v38 = v39;
              v40 = (__int64 **)(*(_QWORD *)(a3 + 152) + 8LL * (*(_QWORD *)(v39 + 8) % *(_QWORD *)(a3 + 160)));
              if ( *v40 )
                break;
              *v40 = v38;
              v31 = (__int64 *)*v31;
              if ( !v31 )
                goto LABEL_41;
            }
          }
        }
LABEL_41:
        *(_BYTE *)(a3 + 208) = *(_BYTE *)(v4 + 208);
      }
      v4 += 216;
      a3 += 216;
    }
    while ( a2 != v4 );
  }
  return a3;
}
