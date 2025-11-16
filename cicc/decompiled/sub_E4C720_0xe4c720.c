// Function: sub_E4C720
// Address: 0xe4c720
//
__int64 __fastcall sub_E4C720(__int64 a1, _QWORD ***a2, unsigned int a3, __m128i *a4)
{
  void (__fastcall *v6)(__m128i *, __m128i *, __int64); // rax
  __m128i v7; // xmm0
  __int64 v8; // rdx
  __m128i v9; // xmm1
  __int64 v10; // rax
  _QWORD **v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD **v15; // r14
  unsigned int v16; // r13d
  __int64 v17; // rsi
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rdi
  _QWORD **v29; // [rsp+8h] [rbp-C8h] BYREF
  __m128i v30; // [rsp+10h] [rbp-C0h] BYREF
  void (__fastcall *v31)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-B0h]
  __int64 v32; // [rsp+28h] [rbp-A8h]
  __int64 v33[4]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v34; // [rsp+50h] [rbp-80h]
  __int64 *v35; // [rsp+60h] [rbp-70h]
  unsigned int v36; // [rsp+70h] [rbp-60h]
  _QWORD *v37; // [rsp+80h] [rbp-50h]
  unsigned int v38; // [rsp+90h] [rbp-40h]

  sub_E48650(v33, a1);
  v6 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a4[1].m128i_i64[0];
  v7 = _mm_loadu_si128(a4);
  v8 = v32;
  v9 = _mm_loadu_si128(&v30);
  a4[1].m128i_i64[0] = 0;
  v31 = v6;
  v10 = a4[1].m128i_i64[1];
  *a4 = v9;
  a4[1].m128i_i64[1] = v8;
  v32 = v10;
  v11 = *a2;
  *a2 = 0;
  v29 = v11;
  v30 = v7;
  v12 = sub_E49E40(v33, (__int64)&v29, a3, &v30);
  v15 = v29;
  v16 = v12;
  if ( v29 )
  {
    sub_BA9C10(v29, (__int64)&v29, v13, v14);
    j_j___libc_free_0(v15, 880);
  }
  if ( v31 )
    v31(&v30, &v30, 3);
  v17 = v38;
  if ( v38 )
  {
    v18 = v37;
    v19 = &v37[2 * v38];
    do
    {
      if ( *v18 != -4096 && *v18 != -8192 )
      {
        v20 = v18[1];
        if ( v20 )
          sub_B91220((__int64)(v18 + 1), v20);
      }
      v18 += 2;
    }
    while ( v19 != v18 );
    v17 = v38;
  }
  sub_C7D6A0((__int64)v37, 16 * v17, 8);
  if ( v36 )
  {
    v24 = sub_1061AC0();
    v25 = sub_1061AD0();
    v26 = v35;
    v27 = v25;
    v22 = v36;
    v21 = (__int64)&v35[v22];
    if ( v35 != &v35[v22] )
    {
      do
      {
        while ( (unsigned __int8)sub_1061B40(*v26, v24) )
        {
          if ( (__int64 *)v21 == ++v26 )
            goto LABEL_21;
        }
        v28 = *v26++;
        sub_1061B40(v28, v27);
      }
      while ( (__int64 *)v21 != v26 );
LABEL_21:
      v21 = (__int64)v35;
      v22 = v36;
    }
  }
  else
  {
    v21 = (__int64)v35;
    v22 = 0;
  }
  sub_C7D6A0(v21, v22 * 8, 8);
  sub_C7D6A0(v33[2], 8LL * v34, 8);
  return v16;
}
