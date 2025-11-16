// Function: sub_39D8FF0
// Address: 0x39d8ff0
//
void __fastcall sub_39D8FF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int32 i; // eax
  __m128i *v6; // rax
  __int64 v7; // rcx
  size_t v8; // rdx
  __m128i *v9; // rdi
  __int64 v10; // rsi
  __m128i *v11; // r8
  __m128i v12; // xmm3
  unsigned __int64 *v13; // [rsp+8h] [rbp-178h]
  __m128i *v14; // [rsp+20h] [rbp-160h]
  __int64 *v15; // [rsp+40h] [rbp-140h]
  __int32 v16; // [rsp+4Ch] [rbp-134h]
  __int32 v17; // [rsp+4Ch] [rbp-134h]
  unsigned __int64 v18[2]; // [rsp+60h] [rbp-120h] BYREF
  _BYTE v19[16]; // [rsp+70h] [rbp-110h] BYREF
  __m128i *v20; // [rsp+80h] [rbp-100h] BYREF
  size_t n; // [rsp+88h] [rbp-F8h]
  __m128i v22; // [rsp+90h] [rbp-F0h] BYREF
  void *v23; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v24; // [rsp+A8h] [rbp-D8h]
  __int64 v25; // [rsp+B0h] [rbp-D0h]
  __int64 v26; // [rsp+B8h] [rbp-C8h]
  int v27; // [rsp+C0h] [rbp-C0h]
  unsigned __int64 *v28; // [rsp+C8h] [rbp-B8h]
  __m128i *p_src; // [rsp+D0h] [rbp-B0h]
  size_t v30; // [rsp+D8h] [rbp-A8h]
  __m128i src; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v32; // [rsp+F0h] [rbp-90h] BYREF
  __m128i v33; // [rsp+100h] [rbp-80h] BYREF
  __int64 v34; // [rsp+110h] [rbp-70h]
  void *dest; // [rsp+118h] [rbp-68h]
  size_t v36; // [rsp+120h] [rbp-60h]
  _QWORD v37[2]; // [rsp+128h] [rbp-58h] BYREF
  __m128i v38; // [rsp+138h] [rbp-48h] BYREF
  int v39; // [rsp+148h] [rbp-38h]
  bool v40; // [rsp+14Ch] [rbp-34h]

  v3 = *(__int64 **)(a3 + 8);
  v15 = *(__int64 **)(a3 + 16);
  if ( v3 != v15 )
  {
    v13 = (unsigned __int64 *)(a2 + 352);
    for ( i = 0; ; i = v16 )
    {
      v17 = i;
      v18[1] = 0;
      v18[0] = (unsigned __int64)v19;
      v19[0] = 0;
      v23 = &unk_49EFBE0;
      v27 = 1;
      v26 = 0;
      v25 = 0;
      v24 = 0;
      v28 = v18;
      if ( *((int *)v3 + 2) < 0 )
        (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)*v3 + 40LL))(*v3, &v23);
      else
        sub_15537D0(*v3, (__int64)&v23, 1, 0);
      v33.m128i_i32[0] = v17;
      v33.m128i_i64[1] = 0;
      dest = v37;
      v34 = 0;
      v36 = 0;
      LOBYTE(v37[0]) = 0;
      v38 = 0u;
      v39 = 0;
      v40 = 0;
      v16 = v17 + 1;
      if ( v26 != v24 )
        sub_16E7BA0((__int64 *)&v23);
      v20 = &v22;
      sub_39CF630((__int64 *)&v20, (_BYTE *)*v28, *v28 + v28[1]);
      v6 = v20;
      p_src = &src;
      if ( v20 == &v22 )
        break;
      v7 = v22.m128i_i64[0];
      v8 = n;
      p_src = v20;
      v20 = &v22;
      v9 = (__m128i *)dest;
      src.m128i_i64[0] = v22.m128i_i64[0];
      v30 = n;
      n = 0;
      v22.m128i_i8[0] = 0;
      v32 = 0u;
      if ( v6 == &src )
        goto LABEL_28;
      if ( dest == v37 )
      {
        dest = v6;
        v36 = v8;
        v37[0] = v7;
      }
      else
      {
        v10 = v37[0];
        dest = v6;
        v36 = v8;
        v37[0] = v7;
        if ( v9 )
        {
          p_src = v9;
          src.m128i_i64[0] = v10;
          goto LABEL_11;
        }
      }
      p_src = &src;
      v9 = &src;
LABEL_11:
      v30 = 0;
      v9->m128i_i8[0] = 0;
      v38 = _mm_load_si128(&v32);
      if ( p_src != &src )
        j_j___libc_free_0((unsigned __int64)p_src);
      if ( v20 != &v22 )
        j_j___libc_free_0((unsigned __int64)v20);
      v11 = *(__m128i **)(a2 + 360);
      v39 = v3[1] & 0x7FFFFFFF;
      v40 = *((int *)v3 + 2) < 0;
      if ( v11 == *(__m128i **)(a2 + 368) )
      {
        sub_39D8CB0(v13, v11, &v33);
      }
      else
      {
        if ( v11 )
        {
          v14 = v11;
          *v11 = _mm_load_si128(&v33);
          v11[1].m128i_i64[0] = v34;
          v11[1].m128i_i64[1] = (__int64)&v11[2].m128i_i64[1];
          sub_39CF630(&v11[1].m128i_i64[1], dest, (__int64)dest + v36);
          *(__m128i *)((char *)v14 + 56) = _mm_loadu_si128(&v38);
          v14[4].m128i_i32[2] = v39;
          v14[4].m128i_i8[12] = v40;
          v11 = *(__m128i **)(a2 + 360);
        }
        *(_QWORD *)(a2 + 360) = v11 + 5;
      }
      if ( dest != v37 )
        j_j___libc_free_0((unsigned __int64)dest);
      sub_16E7BC0((__int64 *)&v23);
      if ( (_BYTE *)v18[0] != v19 )
        j_j___libc_free_0(v18[0]);
      v3 += 2;
      if ( v15 == v3 )
        return;
    }
    v8 = n;
    v9 = (__m128i *)dest;
    n = 0;
    v12 = _mm_load_si128(&v22);
    v32 = 0u;
    v30 = v8;
    v22.m128i_i8[0] = 0;
    src = v12;
LABEL_28:
    if ( v8 )
    {
      if ( v8 == 1 )
        v9->m128i_i8[0] = src.m128i_i8[0];
      else
        memcpy(v9, &src, v8);
      v8 = v30;
      v9 = (__m128i *)dest;
    }
    v36 = v8;
    v9->m128i_i8[v8] = 0;
    v9 = p_src;
    goto LABEL_11;
  }
}
