// Function: sub_149B390
// Address: 0x149b390
//
void __fastcall sub_149B390(__m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  size_t *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  _BYTE *v14; // rdi
  _BYTE *v15; // r8
  size_t v16; // r15
  __int64 v17; // rax
  _BYTE *v18; // [rsp+0h] [rbp-50h]
  size_t *v19; // [rsp+8h] [rbp-48h]
  size_t *v20; // [rsp+8h] [rbp-48h]
  size_t v21; // [rsp+18h] [rbp-38h] BYREF

  a1[7].m128i_i64[0] = 0;
  a1[7].m128i_i64[1] = 0;
  a1[8].m128i_i64[0] = 0;
  a1[8].m128i_i32[2] = 0;
  j___libc_free_0(0);
  v3 = *(unsigned int *)(a2 + 136);
  a1[8].m128i_i32[2] = v3;
  if ( (_DWORD)v3 )
  {
    v7 = sub_22077B0(40 * v3);
    a1[7].m128i_i64[1] = v7;
    v8 = v7;
    a1[8].m128i_i64[0] = *(_QWORD *)(a2 + 128);
    if ( !a1[8].m128i_i32[2] )
      goto LABEL_3;
    v9 = 0;
    v10 = &v21;
    while ( 1 )
    {
      v11 = 40 * v9;
      v12 = 40 * v9 + v8;
      if ( v12 )
      {
        *(_DWORD *)v12 = *(_DWORD *)(*(_QWORD *)(a2 + 120) + 40 * v9);
        v12 = v11 + a1[7].m128i_i64[1];
      }
      if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
        break;
LABEL_6:
      if ( a1[8].m128i_u32[2] <= (unsigned __int64)++v9 )
        goto LABEL_3;
      v8 = a1[7].m128i_i64[1];
    }
    v13 = *(_QWORD *)(a2 + 120) + v11;
    v14 = (_BYTE *)(v12 + 24);
    *(_QWORD *)(v12 + 8) = v12 + 24;
    v15 = *(_BYTE **)(v13 + 8);
    v16 = *(_QWORD *)(v13 + 16);
    if ( &v15[v16] && !v15 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v21 = *(_QWORD *)(v13 + 16);
    if ( v16 > 0xF )
    {
      v18 = v15;
      v19 = v10;
      v17 = sub_22409D0(v12 + 8, v10, 0);
      v10 = v19;
      v15 = v18;
      *(_QWORD *)(v12 + 8) = v17;
      v14 = (_BYTE *)v17;
      *(_QWORD *)(v12 + 24) = v21;
    }
    else
    {
      if ( v16 == 1 )
      {
        *(_BYTE *)(v12 + 24) = *v15;
LABEL_16:
        *(_QWORD *)(v12 + 16) = v16;
        v14[v16] = 0;
        goto LABEL_6;
      }
      if ( !v16 )
        goto LABEL_16;
    }
    v20 = v10;
    memcpy(v14, v15, v16);
    v16 = v21;
    v14 = *(_BYTE **)(v12 + 8);
    v10 = v20;
    goto LABEL_16;
  }
  a1[7].m128i_i64[1] = 0;
  a1[8].m128i_i64[0] = 0;
LABEL_3:
  v4 = *(_BYTE *)(a2 + 146);
  v5 = *(unsigned __int16 *)(a2 + 144);
  a1[9].m128i_i64[1] = 0;
  a1[10].m128i_i64[0] = 0;
  a1[9].m128i_i16[0] = v5;
  a1[9].m128i_i8[2] = v4;
  a1[10].m128i_i64[1] = 0;
  a1[11].m128i_i64[0] = 0;
  a1[11].m128i_i64[1] = 0;
  a1[12].m128i_i64[0] = 0;
  *a1 = _mm_loadu_si128((const __m128i *)a2);
  a1[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
  a1[2] = _mm_loadu_si128((const __m128i *)(a2 + 32));
  a1[3] = _mm_loadu_si128((const __m128i *)(a2 + 48));
  a1[4] = _mm_loadu_si128((const __m128i *)(a2 + 64));
  a1[5] = _mm_loadu_si128((const __m128i *)(a2 + 80));
  a1[6].m128i_i64[0] = *(_QWORD *)(a2 + 96);
  a1[6].m128i_i16[4] = *(_WORD *)(a2 + 104);
  sub_149ABC0((__int64)&a1[9].m128i_i64[1], (const __m128i **)(a2 + 152), v5);
  sub_149ABC0((__int64)a1[11].m128i_i64, (const __m128i **)(a2 + 176), v6);
}
