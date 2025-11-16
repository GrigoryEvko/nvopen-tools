// Function: sub_3111610
// Address: 0x3111610
//
__int64 *__fastcall sub_3111610(__int64 *a1, int a2, __int64 a3)
{
  __m128i *v5; // rdx
  unsigned __int64 *v6; // rax
  void *v7; // rdi
  _BYTE *v8; // r8
  size_t v9; // r13
  size_t v11; // rdx
  unsigned __int8 *v12; // rsi
  __int64 v13; // rax
  __m128i si128; // xmm0
  __m128i v15; // xmm0
  __m128i v16; // xmm0
  __m128i v17; // xmm0
  __int64 v18; // rax
  _BYTE *v19; // [rsp+8h] [rbp-A8h]
  size_t v20; // [rsp+18h] [rbp-98h] BYREF
  unsigned __int64 v21[2]; // [rsp+20h] [rbp-90h] BYREF
  _BYTE v22[16]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v23[3]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v24; // [rsp+58h] [rbp-58h]
  __m128i *v25; // [rsp+60h] [rbp-50h]
  __int64 v26; // [rsp+68h] [rbp-48h]
  unsigned __int64 *v27; // [rsp+70h] [rbp-40h]

  v26 = 0x100000000LL;
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0;
  v23[0] = &unk_49DD210;
  v22[0] = 0;
  v23[1] = 0;
  v23[2] = 0;
  v24 = 0;
  v25 = 0;
  v27 = v21;
  sub_CB5980((__int64)v23, 0, 0, 0);
  v5 = v25;
  switch ( a2 )
  {
    case 0:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 6 )
      {
        sub_CB6200((__int64)v23, (unsigned __int8 *)"success", 7u);
      }
      else
      {
        v25->m128i_i32[0] = 1667462515;
        v5->m128i_i16[2] = 29541;
        v5->m128i_i8[6] = 115;
        v25 = (__m128i *)((char *)v25 + 7);
      }
      break;
    case 1:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0xA )
      {
        sub_CB6200((__int64)v23, "end of File", 0xBu);
      }
      else
      {
        v25->m128i_i8[10] = 101;
        qmemcpy(v5, "end of Fil", 10);
        v25 = (__m128i *)((char *)v25 + 11);
      }
      break;
    case 2:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0x1F )
      {
        sub_CB6200((__int64)v23, "invalid codegen data (bad magic)", 0x20u);
      }
      else
      {
        *v25 = _mm_load_si128((const __m128i *)&xmmword_44CF3D0);
        v5[1] = _mm_load_si128((const __m128i *)&xmmword_42AE620);
        v25 += 2;
      }
      break;
    case 3:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0x2C )
      {
        sub_CB6200((__int64)v23, "invalid codegen data (file header is corrupt)", 0x2Du);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44CF3D0);
        v25[2].m128i_i32[2] = 1953527154;
        v5[2].m128i_i64[0] = 0x726F632073692072LL;
        *v5 = si128;
        v15 = _mm_load_si128((const __m128i *)&xmmword_42AE630);
        v5[2].m128i_i8[12] = 41;
        v5[1] = v15;
        v25 = (__m128i *)((char *)v25 + 45);
      }
      break;
    case 4:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0x11 )
      {
        sub_CB6200((__int64)v23, "empty codegen data", 0x12u);
      }
      else
      {
        v16 = _mm_load_si128((const __m128i *)&xmmword_44CF3E0);
        v25[1].m128i_i16[0] = 24948;
        *v5 = v16;
        v25 = (__m128i *)((char *)v25 + 18);
      }
      break;
    case 5:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0x15 )
      {
        sub_CB6200((__int64)v23, "malformed codegen data", 0x16u);
      }
      else
      {
        v17 = _mm_load_si128((const __m128i *)&xmmword_44CF3F0);
        v25[1].m128i_i32[0] = 1633951854;
        v5[1].m128i_i16[2] = 24948;
        *v5 = v17;
        v25 = (__m128i *)((char *)v25 + 22);
      }
      break;
    case 6:
      if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 0x1F )
      {
        sub_CB6200((__int64)v23, "unsupported codegen data version", 0x20u);
      }
      else
      {
        *v25 = _mm_load_si128((const __m128i *)&xmmword_44CF400);
        v5[1] = _mm_load_si128((const __m128i *)&xmmword_44CF410);
        v25 += 2;
      }
      break;
    default:
      break;
  }
  if ( *(_QWORD *)(a3 + 8) )
  {
    if ( (unsigned __int64)(v24 - (_QWORD)v25) <= 1 )
    {
      v18 = sub_CB6200((__int64)v23, (unsigned __int8 *)": ", 2u);
      sub_CB6200(v18, *(unsigned __int8 **)a3, *(_QWORD *)(a3 + 8));
    }
    else
    {
      v25->m128i_i16[0] = 8250;
      v11 = *(_QWORD *)(a3 + 8);
      v12 = *(unsigned __int8 **)a3;
      v25 = (__m128i *)((char *)v25 + 2);
      sub_CB6200((__int64)v23, v12, v11);
    }
  }
  v6 = v27;
  v7 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  v8 = (_BYTE *)*v6;
  v9 = v6[1];
  if ( v9 + *v6 && !v8 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v20 = v6[1];
  if ( v9 > 0xF )
  {
    v19 = v8;
    v13 = sub_22409D0((__int64)a1, &v20, 0);
    v8 = v19;
    *a1 = v13;
    v7 = (void *)v13;
    a1[2] = v20;
LABEL_17:
    memcpy(v7, v8, v9);
    v9 = v20;
    v7 = (void *)*a1;
    goto LABEL_9;
  }
  if ( v9 == 1 )
  {
    *((_BYTE *)a1 + 16) = *v8;
    goto LABEL_9;
  }
  if ( v9 )
    goto LABEL_17;
LABEL_9:
  a1[1] = v9;
  *((_BYTE *)v7 + v9) = 0;
  v23[0] = &unk_49DD210;
  sub_CB5840((__int64)v23);
  if ( (_BYTE *)v21[0] != v22 )
    j_j___libc_free_0(v21[0]);
  return a1;
}
