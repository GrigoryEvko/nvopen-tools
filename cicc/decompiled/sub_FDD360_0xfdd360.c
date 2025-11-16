// Function: sub_FDD360
// Address: 0xfdd360
//
__int64 __fastcall sub_FDD360(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r13
  const char *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  unsigned __int8 *v11; // rsi
  _BYTE *v12; // rax
  size_t v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  size_t v17; // rdx
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  void *v20; // rdx
  unsigned __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // r15
  unsigned __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rax
  size_t v30; // rdx
  __int64 v31; // rax
  size_t v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // r13
  __int64 v35; // r15
  __int64 v36; // rdx
  char *v37; // rsi
  __m128i *v38; // rdx
  __m128i v39; // xmm0
  __int64 v40; // rdi
  void *v41; // rdx
  __int64 v42; // rdi
  __int64 i; // [rsp+18h] [rbp-78h]
  int v44; // [rsp+3Ch] [rbp-54h] BYREF
  unsigned __int8 *v45; // [rsp+40h] [rbp-50h] BYREF
  size_t v46; // [rsp+48h] [rbp-48h]
  _QWORD v47[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 128) )
  {
    v3 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x15u )
    {
      v6 = sub_CB6200(a2, "block-frequency-info: ", 0x16u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CAE0);
      v3[1].m128i_i32[0] = 1868983913;
      v6 = a2;
      v3[1].m128i_i16[2] = 8250;
      *v3 = si128;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v8 = sub_BD5D20(*(_QWORD *)(a1 + 128));
    v10 = *(_BYTE **)(v6 + 32);
    v11 = (unsigned __int8 *)v8;
    v12 = *(_BYTE **)(v6 + 24);
    v13 = v9;
    if ( v9 > v12 - v10 )
    {
      v6 = sub_CB6200(v6, v11, v9);
      v12 = *(_BYTE **)(v6 + 24);
      v10 = *(_BYTE **)(v6 + 32);
    }
    else if ( v9 )
    {
      memcpy(v10, v11, v9);
      v12 = *(_BYTE **)(v6 + 24);
      v10 = (_BYTE *)(v13 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v10;
    }
    if ( v12 == v10 )
    {
      sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v6 + 32);
    }
    v14 = *(_QWORD *)(a1 + 128);
    v15 = *(_QWORD *)(a2 + 32);
    v16 = *(_QWORD *)(v14 + 80);
    for ( i = v14 + 72; i != v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      while ( 1 )
      {
        v34 = v16 - 24;
        if ( !v16 )
          v34 = 0;
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v15) <= 2 )
        {
          v35 = sub_CB6200(a2, (unsigned __int8 *)" - ", 3u);
        }
        else
        {
          *(_BYTE *)(v15 + 2) = 32;
          v35 = a2;
          *(_WORD *)v15 = 11552;
          *(_QWORD *)(a2 + 32) += 3LL;
        }
        v37 = (char *)sub_BD5D20(v34);
        v45 = (unsigned __int8 *)v47;
        if ( v37 )
        {
          sub_FDB1F0((__int64 *)&v45, v37, (__int64)&v37[v36]);
          v17 = v46;
          v18 = v45;
        }
        else
        {
          LOBYTE(v47[0]) = 0;
          v17 = 0;
          v46 = 0;
          v18 = (unsigned __int8 *)v47;
        }
        v19 = sub_CB6200(v35, v18, v17);
        v20 = *(void **)(v19 + 32);
        if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 9u )
        {
          sub_CB6200(v19, ": float = ", 0xAu);
        }
        else
        {
          qmemcpy(v20, ": float = ", 10);
          *(_QWORD *)(v19 + 32) += 10LL;
        }
        if ( v45 != (unsigned __int8 *)v47 )
          j_j___libc_free_0(v45, v47[0] + 1LL);
        LODWORD(v45) = sub_FDD0F0(a1, v34);
        v21 = sub_FE8AC0(a1, &v45);
        v23 = sub_F04D90(a2, v21, v22, 64, 5u);
        v24 = *(_QWORD **)(v23 + 32);
        v25 = v23;
        if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 7u )
        {
          v25 = sub_CB6200(v23, ", int = ", 8u);
        }
        else
        {
          *v24 = 0x203D20746E69202CLL;
          *(_QWORD *)(v23 + 32) += 8LL;
        }
        LODWORD(v45) = sub_FDD0F0(a1, v34);
        v26 = sub_FE8720(a1, &v45);
        sub_CB59D0(v25, v26);
        v27 = sub_FDD0F0(a1, v34);
        v28 = *(_QWORD *)(a1 + 128);
        v44 = v27;
        v29 = sub_FE8990(a1, v28, &v44, 0);
        v46 = v30;
        v45 = (unsigned __int8 *)v29;
        if ( (_BYTE)v30 )
        {
          v41 = *(void **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v41 <= 9u )
          {
            v42 = sub_CB6200(a2, ", count = ", 0xAu);
          }
          else
          {
            v42 = a2;
            qmemcpy(v41, ", count = ", 10);
            *(_QWORD *)(a2 + 32) += 10LL;
          }
          sub_CB59D0(v42, (unsigned __int64)v45);
        }
        v31 = sub_AA5EE0(v34);
        v46 = v32;
        v45 = (unsigned __int8 *)v31;
        if ( (_BYTE)v32 )
        {
          v38 = *(__m128i **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v38 <= 0x1Au )
          {
            v40 = sub_CB6200(a2, ", irr_loop_header_weight = ", 0x1Bu);
          }
          else
          {
            v39 = _mm_load_si128((const __m128i *)&xmmword_3F8CAF0);
            v40 = a2;
            qmemcpy(&v38[1], "r_weight = ", 11);
            *v38 = v39;
            *(_QWORD *)(a2 + 32) += 27LL;
          }
          sub_CB59D0(v40, (unsigned __int64)v45);
        }
        v33 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v33 )
          break;
        *v33 = 10;
        v15 = *(_QWORD *)(a2 + 32) + 1LL;
        *(_QWORD *)(a2 + 32) = v15;
        v16 = *(_QWORD *)(v16 + 8);
        if ( i == v16 )
          goto LABEL_33;
      }
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      v15 = *(_QWORD *)(a2 + 32);
    }
LABEL_33:
    if ( *(_QWORD *)(a2 + 24) == v15 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *(_BYTE *)v15 = 10;
      ++*(_QWORD *)(a2 + 32);
    }
  }
  return a2;
}
