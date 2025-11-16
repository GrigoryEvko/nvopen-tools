// Function: sub_23B0AE0
// Address: 0x23b0ae0
//
void __fastcall sub_23B0AE0(__int64 a1, _BYTE *a2, unsigned __int64 a3)
{
  __m128i *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  unsigned __int8 *v7; // rdi
  __m128i v8; // xmm0
  __m128i *v9; // [rsp+10h] [rbp-130h] BYREF
  __int64 v10; // [rsp+18h] [rbp-128h]
  __m128i v11; // [rsp+20h] [rbp-120h] BYREF
  unsigned __int8 *v12; // [rsp+30h] [rbp-110h] BYREF
  size_t v13; // [rsp+38h] [rbp-108h]
  __int64 v14; // [rsp+40h] [rbp-100h]
  _BYTE v15[24]; // [rsp+48h] [rbp-F8h] BYREF
  _QWORD v16[3]; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v17; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v18; // [rsp+88h] [rbp-B8h]
  unsigned __int8 **v19; // [rsp+90h] [rbp-B0h]
  _QWORD v20[3]; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i v21; // [rsp+B8h] [rbp-88h] BYREF
  void *v22; // [rsp+C8h] [rbp-78h] BYREF
  __m128i *v23; // [rsp+D0h] [rbp-70h]
  __int64 v24; // [rsp+D8h] [rbp-68h]
  __m128i v25; // [rsp+E0h] [rbp-60h] BYREF
  _QWORD v26[2]; // [rsp+F0h] [rbp-50h] BYREF
  _QWORD v27[8]; // [rsp+100h] [rbp-40h] BYREF

  sub_23AF980((__int64)&v9, a2, a3);
  v4 = v9;
  if ( v9 == &v11 )
  {
    v8 = _mm_load_si128(&v11);
    v6 = v10;
    v11.m128i_i8[0] = 0;
    v10 = 0;
    v21 = v8;
  }
  else
  {
    v5 = v11.m128i_i64[0];
    v6 = v10;
    v9 = &v11;
    v11.m128i_i8[0] = 0;
    v21.m128i_i64[0] = v5;
    v10 = 0;
    if ( v4 != &v21 )
    {
      v17.m128i_i64[0] = v5;
      goto LABEL_4;
    }
  }
  v4 = &v17;
  v17 = _mm_loadu_si128(&v21);
LABEL_4:
  v20[1] = 35;
  v20[0] = "  <a>{0}. {1} invalidated</a><br/>\n";
  v20[2] = v27;
  v21.m128i_i64[0] = 2;
  v21.m128i_i8[8] = 1;
  v22 = &unk_49E64B0;
  v23 = &v25;
  if ( v4 == &v17 )
  {
    v25 = _mm_loadu_si128(&v17);
  }
  else
  {
    v23 = v4;
    v25.m128i_i64[0] = v17.m128i_i64[0];
  }
  v24 = v6;
  v26[1] = a1 + 36;
  v18 = 0x100000000LL;
  v26[0] = &unk_49DB138;
  v27[0] = v26;
  v27[1] = &v22;
  v16[0] = &unk_49DD288;
  v19 = &v12;
  v12 = v15;
  v13 = 0;
  v14 = 20;
  v16[1] = 2;
  v16[2] = 0;
  v17 = 0u;
  sub_CB5980((__int64)v16, 0, 0, 0);
  sub_CB6840((__int64)v16, (__int64)v20);
  v16[0] = &unk_49DD388;
  sub_CB5840((__int64)v16);
  v22 = &unk_49E64B0;
  if ( v23 != &v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  if ( v9 != &v11 )
    j_j___libc_free_0((unsigned __int64)v9);
  sub_CB6200(*(_QWORD *)(a1 + 40), v12, v13);
  v7 = v12;
  ++*(_DWORD *)(a1 + 36);
  if ( v7 != v15 )
    _libc_free((unsigned __int64)v7);
}
