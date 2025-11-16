// Function: sub_3021B30
// Address: 0x3021b30
//
void __fastcall sub_3021B30(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rax
  __int64 v6; // rdx
  char v7; // al
  __m128i *v8; // rdx
  bool v9; // zf
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v12; // xmm0
  unsigned __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int8 *v14; // [rsp+10h] [rbp-40h] BYREF
  size_t v15; // [rsp+18h] [rbp-38h]
  _BYTE v16[48]; // [rsp+20h] [rbp-30h] BYREF

  v2 = a2;
  v14 = v16;
  v15 = 0;
  v16[0] = 0;
  if ( (unsigned __int8)sub_CE9F90(a1, &v13) )
  {
    v3 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x13u )
    {
      v2 = sub_CB6200(a2, (unsigned __int8 *)".attribute(.unified(", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43270F0);
      v3[1].m128i_i32[0] = 677668201;
      *v3 = si128;
      *(_QWORD *)(a2 + 32) += 20LL;
    }
    v5 = sub_CB59D0(v2, v13);
  }
  else
  {
    v7 = sub_CE9650(a1, &v14);
    v8 = *(__m128i **)(a2 + 32);
    v9 = v7 == 0;
    v10 = *(_QWORD *)(a2 + 24);
    if ( v9 )
    {
      if ( (unsigned __int64)(v10 - (_QWORD)v8) <= 0x14 )
      {
        sub_CB6200(a2, ".attribute(.unified) ", 0x15u);
      }
      else
      {
        v12 = _mm_load_si128((const __m128i *)&xmmword_43270F0);
        v8[1].m128i_i32[0] = 694445417;
        v8[1].m128i_i8[4] = 32;
        *v8 = v12;
        *(_QWORD *)(a2 + 32) += 21LL;
      }
      goto LABEL_7;
    }
    if ( (unsigned __int64)(v10 - (_QWORD)v8) <= 0x13 )
    {
      v2 = sub_CB6200(a2, (unsigned __int8 *)".attribute(.unified(", 0x14u);
    }
    else
    {
      v11 = _mm_load_si128((const __m128i *)&xmmword_43270F0);
      v8[1].m128i_i32[0] = 677668201;
      *v8 = v11;
      *(_QWORD *)(a2 + 32) += 20LL;
    }
    v5 = sub_CB6200(v2, v14, v15);
  }
  v6 = *(_QWORD *)(v5 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v6) <= 2 )
  {
    sub_CB6200(v5, (unsigned __int8 *)")) ", 3u);
  }
  else
  {
    *(_BYTE *)(v6 + 2) = 32;
    *(_WORD *)v6 = 10537;
    *(_QWORD *)(v5 + 32) += 3LL;
  }
LABEL_7:
  if ( v14 != v16 )
    j_j___libc_free_0((unsigned __int64)v14);
}
