// Function: sub_33FB620
// Address: 0x33fb620
//
unsigned __int8 *__fastcall sub_33FB620(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8)
{
  unsigned __int16 v10; // bx
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  char v17; // si
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  __m128i v22; // xmm0
  _DWORD *v23; // rbx
  unsigned __int16 v24; // si
  unsigned int v25; // eax
  char v26; // [rsp+7h] [rbp-79h]
  unsigned __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+10h] [rbp-70h] BYREF
  __int64 v29; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+30h] [rbp-50h]
  __int64 v33; // [rsp+38h] [rbp-48h]
  __m128i v34; // [rsp+40h] [rbp-40h] BYREF

  v10 = a5;
  v11 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v28 = a5;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v29 = a6;
  if ( v12 == (_WORD)a5 )
  {
    if ( (_WORD)a5 || v13 == v29 )
      return sub_33FAF80(a1, 216, a4, (unsigned int)v28, v29, a6, a7);
    v34.m128i_i64[1] = v13;
    v34.m128i_i16[0] = 0;
LABEL_5:
    v32 = sub_3007260((__int64)&v34);
    v15 = v32;
    v33 = v16;
    v17 = v16;
    if ( !v10 )
      goto LABEL_6;
    goto LABEL_18;
  }
  v34.m128i_i16[0] = v12;
  v34.m128i_i64[1] = v13;
  if ( !v12 )
    goto LABEL_5;
  if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
    goto LABEL_30;
  v15 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
  v17 = byte_444C4A0[16 * v12 - 8];
  if ( (_WORD)a5 )
  {
LABEL_18:
    if ( v10 != 1 && (unsigned __int16)(v10 - 504) > 7u )
    {
      v21 = *(_QWORD *)&byte_444C4A0[16 * v10 - 16];
      if ( !byte_444C4A0[16 * v10 - 8] )
        goto LABEL_21;
      goto LABEL_7;
    }
LABEL_30:
    BUG();
  }
LABEL_6:
  v27 = v15;
  v18 = sub_3007260((__int64)&v28);
  v15 = v27;
  v19 = v18;
  v31 = v20;
  LOBYTE(v18) = v20;
  v21 = v19;
  v30 = v19;
  if ( !(_BYTE)v18 )
    goto LABEL_21;
LABEL_7:
  if ( !v17 )
    goto LABEL_8;
LABEL_21:
  if ( v21 <= v15 )
    return sub_33FAF80(a1, 216, a4, (unsigned int)v28, v29, a6, a7);
LABEL_8:
  v22 = _mm_loadu_si128((const __m128i *)&a8);
  v23 = *(_DWORD **)(a1 + 16);
  v34 = v22;
  if ( !(_WORD)a8 )
  {
    v26 = sub_3007030((__int64)&v34);
    if ( sub_30070B0((__int64)&v34) )
      goto LABEL_27;
    if ( !v26 )
      goto LABEL_12;
LABEL_25:
    v25 = v23[16];
    goto LABEL_13;
  }
  v24 = a8 - 17;
  if ( (unsigned __int16)(a8 - 10) > 6u && (unsigned __int16)(a8 - 126) > 0x31u )
  {
    if ( v24 > 0xD3u )
    {
LABEL_12:
      v25 = v23[15];
      goto LABEL_13;
    }
    goto LABEL_27;
  }
  if ( v24 > 0xD3u )
    goto LABEL_25;
LABEL_27:
  v25 = v23[17];
LABEL_13:
  if ( v25 > 2 )
    goto LABEL_30;
  return sub_33FAF80(a1, 215 - v25, a4, (unsigned int)v28, v29, a6, v22);
}
