// Function: sub_23B0DB0
// Address: 0x23b0db0
//
void __fastcall sub_23B0DB0(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4)
{
  __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdi
  unsigned __int8 *v9; // rdi
  __m128i v10; // xmm0
  __m128i *v11; // [rsp+10h] [rbp-160h] BYREF
  __int64 v12; // [rsp+18h] [rbp-158h]
  __m128i v13; // [rsp+20h] [rbp-150h] BYREF
  unsigned __int8 *v14; // [rsp+30h] [rbp-140h] BYREF
  size_t v15; // [rsp+38h] [rbp-138h]
  __int64 v16; // [rsp+40h] [rbp-130h]
  _BYTE v17[24]; // [rsp+48h] [rbp-128h] BYREF
  _QWORD v18[5]; // [rsp+60h] [rbp-110h] BYREF
  __m128i v19; // [rsp+88h] [rbp-E8h] BYREF
  void *v20; // [rsp+98h] [rbp-D8h]
  __int64 v21; // [rsp+A0h] [rbp-D0h]
  _QWORD v22[3]; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v23; // [rsp+C8h] [rbp-A8h] BYREF
  _QWORD v24[2]; // [rsp+D8h] [rbp-98h] BYREF
  void *v25; // [rsp+E8h] [rbp-88h] BYREF
  __m128i *v26; // [rsp+F0h] [rbp-80h]
  __int64 v27; // [rsp+F8h] [rbp-78h]
  __m128i v28; // [rsp+100h] [rbp-70h] BYREF
  _QWORD v29[2]; // [rsp+110h] [rbp-60h] BYREF
  _QWORD v30[10]; // [rsp+120h] [rbp-50h] BYREF

  sub_23AF980((__int64)&v11, a2, a3);
  v6 = v11;
  if ( v11 == &v13 )
  {
    v10 = _mm_load_si128(&v13);
    v8 = v12;
    v13.m128i_i8[0] = 0;
    v12 = 0;
    v23 = v10;
  }
  else
  {
    v7 = v13.m128i_i64[0];
    v8 = v12;
    v11 = &v13;
    v13.m128i_i8[0] = 0;
    v23.m128i_i64[0] = v7;
    v12 = 0;
    if ( v6 != &v23 )
    {
      v19.m128i_i64[0] = v7;
      goto LABEL_4;
    }
  }
  v6 = &v19;
  v19 = _mm_loadu_si128(&v23);
LABEL_4:
  v24[0] = &unk_49E6618;
  v21 = a1 + 36;
  v20 = &unk_49DB138;
  v22[0] = "  <a>{0}. {1} on {2} ignored</a><br/>\n";
  v22[1] = 38;
  v22[2] = v30;
  v23.m128i_i64[0] = 3;
  v23.m128i_i8[8] = 1;
  v24[1] = a4;
  v25 = &unk_49E64B0;
  v26 = &v28;
  if ( v6 == &v19 )
  {
    v28 = _mm_loadu_si128(&v19);
  }
  else
  {
    v26 = v6;
    v28.m128i_i64[0] = v19.m128i_i64[0];
  }
  v29[1] = a1 + 36;
  v29[0] = &unk_49DB138;
  v30[0] = v29;
  v30[1] = &v25;
  v27 = v8;
  v30[2] = v24;
  v19.m128i_i64[0] = 0x100000000LL;
  v18[0] = &unk_49DD288;
  v19.m128i_i64[1] = (__int64)&v14;
  v14 = v17;
  v15 = 0;
  v16 = 20;
  v18[1] = 2;
  memset(&v18[2], 0, 24);
  sub_CB5980((__int64)v18, 0, 0, 0);
  sub_CB6840((__int64)v18, (__int64)v22);
  v18[0] = &unk_49DD388;
  sub_CB5840((__int64)v18);
  v25 = &unk_49E64B0;
  if ( v26 != &v28 )
    j_j___libc_free_0((unsigned __int64)v26);
  if ( v11 != &v13 )
    j_j___libc_free_0((unsigned __int64)v11);
  sub_CB6200(*(_QWORD *)(a1 + 40), v14, v15);
  v9 = v14;
  ++*(_DWORD *)(a1 + 36);
  if ( v9 != v17 )
    _libc_free((unsigned __int64)v9);
}
