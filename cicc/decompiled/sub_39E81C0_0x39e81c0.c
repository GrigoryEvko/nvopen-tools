// Function: sub_39E81C0
// Address: 0x39e81c0
//
__m128i *__fastcall sub_39E81C0(
        __int64 a1,
        unsigned __int64 *a2,
        char a3,
        char a4,
        __int64 a5,
        __int64 *a6,
        _DWORD **a7,
        char a8)
{
  unsigned __int64 v8; // r14
  __int64 v9; // r13
  _DWORD *v10; // rbx
  __m128i *v11; // rax
  __m128i *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdi
  __int8 v18; // r14
  __int64 (__fastcall *v20)(__int64); // rax
  __int64 *v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h] BYREF
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  _DWORD *v34; // [rsp+38h] [rbp-38h] BYREF

  v8 = *a2;
  *a2 = 0;
  v9 = *a6;
  *a6 = 0;
  v10 = *a7;
  *a7 = 0;
  v11 = (__m128i *)sub_22077B0(0x2B0u);
  v12 = v11;
  if ( v11 )
  {
    sub_38DCAE0(v11, a1);
    v13 = *(_QWORD *)(a1 + 16);
    v12[16].m128i_i64[1] = v8;
    v12->m128i_i64[0] = (__int64)off_4A40718;
    v12[17].m128i_i64[1] = v13;
    v12[17].m128i_i64[0] = v8;
    v12[18].m128i_i64[0] = a5;
    v14 = 0;
    if ( v10 )
    {
      sub_390A0A0((__int64)&v31, v10, (__int64)v12[40].m128i_i64);
      v14 = v31;
    }
    v33 = v9;
    v34 = v10;
    v32 = v14;
    v31 = 0;
    v15 = sub_22077B0(0x850u);
    v16 = v15;
    if ( v15 )
      sub_390A820(v15, a1, &v34, &v33, &v32);
    v17 = v32;
    v12[18].m128i_i64[1] = v16;
    if ( v17 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
    if ( v33 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
    if ( v34 )
      (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v34 + 8LL))(v34);
    if ( v31 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
    v12[19].m128i_i64[0] = (__int64)v12[20].m128i_i64;
    v12[19].m128i_i64[1] = 0x8000000000LL;
    v12[28].m128i_i64[1] = 0x8000000000LL;
    v12[28].m128i_i64[0] = (__int64)v12[29].m128i_i64;
    v12[39].m128i_i32[0] = 1;
    v12[37].m128i_i64[0] = (__int64)&unk_49EFC48;
    v12[39].m128i_i64[1] = (__int64)v12[28].m128i_i64;
    v12[38].m128i_i64[1] = 0;
    v12[38].m128i_i64[0] = 0;
    v12[37].m128i_i64[1] = 0;
    sub_16E7A40((__int64)v12[37].m128i_i64, 0, 0, 0);
    v12[42].m128i_i32[0] = 0;
    v12[41].m128i_i64[1] = 0;
    v12[41].m128i_i64[0] = 0;
    v12[40].m128i_i64[1] = 0;
    v12[40].m128i_i64[0] = (__int64)&unk_49EFCB8;
    v18 = v12[42].m128i_i8[8] & 0xF8 | ((4 * a4) | (2 * a8) | a3) & 7;
    v12[42].m128i_i8[8] = v18;
    if ( (v18 & 1) != 0 )
      *(_QWORD *)(v12[18].m128i_i64[0] + 8) = v12 + 37;
  }
  else
  {
    if ( v10 )
      (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 8LL))(v10);
    if ( v9 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
    if ( v8 )
    {
      v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL);
      if ( v20 == sub_16BE0F0 )
      {
        *(_QWORD *)v8 = &unk_49EF340;
        if ( *(_QWORD *)(v8 + 24) != *(_QWORD *)(v8 + 8) )
          sub_16E7BA0((__int64 *)v8);
        v21 = *(__int64 **)(v8 + 40);
        if ( v21 )
        {
          v22 = *(_QWORD *)(v8 + 8);
          if ( !*(_DWORD *)(v8 + 32) || v22 )
          {
            v24 = *(_QWORD *)(v8 + 16) - v22;
          }
          else
          {
            v23 = sub_16E7720();
            v21 = *(__int64 **)(v8 + 40);
            v24 = v23;
          }
          v25 = v21[3];
          v26 = v21[1];
          if ( v24 )
          {
            if ( v26 != v25 )
              sub_16E7BA0(v21);
            v27 = sub_2207820(v24);
            sub_16E7A40((__int64)v21, v27, v24, 1);
          }
          else
          {
            if ( v26 != v25 )
              sub_16E7BA0(v21);
            sub_16E7A40((__int64)v21, 0, 0, 0);
          }
        }
        sub_16E7960(v8);
        j_j___libc_free_0(v8);
      }
      else
      {
        v20(v8);
      }
    }
  }
  return v12;
}
