// Function: sub_B81BF0
// Address: 0xb81bf0
//
__int64 __fastcall sub_B81BF0(__int64 a1, __int64 a2, const void *a3, size_t a4, int a5)
{
  __int64 v6; // rdi
  __int64 result; // rax
  char *v10; // rsi
  _BYTE *v13; // rdi
  char **v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  char *v18; // rax
  size_t v19; // rdx
  void *v20; // rdi
  __int64 v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i si128; // xmm0
  __int64 v25; // rax
  __m128i *v26; // rdx
  __m128i v27; // xmm0
  _QWORD *v28; // [rsp+8h] [rbp-B8h]
  char **v29; // [rsp+18h] [rbp-A8h]
  _QWORD *v30; // [rsp+18h] [rbp-A8h]
  size_t v31; // [rsp+18h] [rbp-A8h]
  _BYTE *v32; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-98h]
  _BYTE v34[144]; // [rsp+30h] [rbp-90h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v32 = v34;
  result = 0xC00000000LL;
  v33 = 0xC00000000LL;
  if ( v6 )
  {
    v10 = (char *)&v32;
    result = (__int64)sub_B809B0(v6, (__int64)&v32, a2);
    if ( (int)qword_4F81B88 > 3 )
    {
      v13 = v32;
      if ( !(_DWORD)v33 )
      {
LABEL_7:
        if ( v13 != v34 )
          return _libc_free(v13, v10);
        return result;
      }
      v15 = sub_C5F790(v32);
      v16 = *(_QWORD *)(v15 + 32);
      v17 = v15;
      if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 5 )
      {
        v17 = sub_CB6200(v15, " -*- '", 6);
      }
      else
      {
        *(_DWORD *)v16 = 757738784;
        *(_WORD *)(v16 + 4) = 10016;
        *(_QWORD *)(v15 + 32) += 6LL;
      }
      v30 = (_QWORD *)v17;
      v18 = (char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
      v10 = v18;
      v20 = (void *)v30[4];
      if ( v19 > v30[3] - (_QWORD)v20 )
      {
        v20 = v30;
        sub_CB6200(v30, v18, v19);
      }
      else if ( v19 )
      {
        v28 = v30;
        v31 = v19;
        memcpy(v20, v18, v19);
        v28[4] += v31;
      }
      v21 = sub_C5F790(v20);
      v22 = *(__m128i **)(v21 + 32);
      v23 = v21;
      if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 0x2Eu )
      {
        v10 = "' is the last user of following pass instances.";
        sub_CB6200(v21, "' is the last user of following pass instances.", 47);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F552B0);
        qmemcpy(&v22[2], "pass instances.", 15);
        *v22 = si128;
        v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F552C0);
        *(_QWORD *)(v21 + 32) += 47LL;
      }
      v25 = sub_C5F790(v23);
      v26 = *(__m128i **)(v25 + 32);
      if ( *(_QWORD *)(v25 + 24) - (_QWORD)v26 <= 0x15u )
      {
        v10 = " Free these instances\n";
        sub_CB6200(v25, " Free these instances\n", 22);
      }
      else
      {
        v27 = _mm_load_si128((const __m128i *)&xmmword_3F552D0);
        v26[1].m128i_i32[0] = 1701015137;
        v26[1].m128i_i16[2] = 2675;
        *v26 = v27;
        *(_QWORD *)(v25 + 32) += 22LL;
      }
    }
    v13 = v32;
    result = (__int64)&v32[8 * (unsigned int)v33];
    v29 = (char **)result;
    if ( (_BYTE *)result != v32 )
    {
      v14 = (char **)v32;
      do
      {
        v10 = *v14++;
        result = sub_B81AB0(a1, v10, a3, a4, a5);
      }
      while ( v29 != v14 );
      v13 = v32;
    }
    goto LABEL_7;
  }
  return result;
}
