// Function: sub_2F19560
// Address: 0x2f19560
//
void __fastcall sub_2F19560(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  _QWORD *v3; // rbx
  __int32 v5; // r14d
  __m128i *v6; // rdi
  __m128i v7; // xmm2
  unsigned __int64 v8; // rsi
  __m128i *v9; // rax
  __int64 v10; // rdx
  size_t v11; // r14
  __m128i v12; // xmm0
  char v13; // al
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  unsigned __int64 *v16; // [rsp+0h] [rbp-180h]
  _QWORD *v17; // [rsp+20h] [rbp-160h]
  __int32 v18; // [rsp+2Ch] [rbp-154h]
  __m128i *v19; // [rsp+50h] [rbp-130h] BYREF
  size_t n; // [rsp+58h] [rbp-128h]
  __m128i v21; // [rsp+60h] [rbp-120h] BYREF
  __m128i *v22; // [rsp+70h] [rbp-110h]
  __int64 v23; // [rsp+78h] [rbp-108h]
  __m128i v24; // [rsp+80h] [rbp-100h] BYREF
  __m128i *v25; // [rsp+90h] [rbp-F0h]
  size_t v26; // [rsp+98h] [rbp-E8h]
  __m128i v27; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v28; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD v29[8]; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v30; // [rsp+100h] [rbp-80h] BYREF
  __int64 v31; // [rsp+110h] [rbp-70h]
  __m128i *p_dest; // [rsp+118h] [rbp-68h]
  size_t v33; // [rsp+120h] [rbp-60h]
  __m128i dest; // [rsp+128h] [rbp-58h] BYREF
  __m128i v35; // [rsp+138h] [rbp-48h] BYREF
  __int16 v36; // [rsp+148h] [rbp-38h]
  char v37; // [rsp+14Ah] [rbp-36h]

  v3 = *(_QWORD **)(a3 + 8);
  v17 = *(_QWORD **)(a3 + 16);
  if ( v3 != v17 )
  {
    v5 = 0;
    v16 = a2 + 56;
    while ( 1 )
    {
      n = 0;
      v19 = &v21;
      v29[5] = 0x100000000LL;
      v21.m128i_i8[0] = 0;
      v29[0] = &unk_49DD210;
      memset(&v29[1], 0, 32);
      v29[6] = &v19;
      sub_CB5980((__int64)v29, 0, 0, 0);
      if ( *((_BYTE *)v3 + 9) )
        (*(void (__fastcall **)(_QWORD, _QWORD *))(*(_QWORD *)*v3 + 48LL))(*v3, v29);
      else
        sub_A5BF40((unsigned __int8 *)*v3, (__int64)v29, 1, 0);
      v30.m128i_i64[1] = 0;
      v18 = v5 + 1;
      v9 = v19;
      v31 = 0;
      p_dest = &dest;
      v33 = 0;
      dest.m128i_i8[0] = 0;
      v35 = 0u;
      HIBYTE(v36) = 0;
      v37 = 0;
      v30.m128i_i32[0] = v5;
      if ( v19 == &v21 )
        break;
      v10 = v21.m128i_i64[0];
      v21.m128i_i8[0] = 0;
      v11 = n;
      n = 0;
      v19 = &v21;
      v24.m128i_i64[0] = v10;
      v25 = &v27;
      if ( v9 == &v24 )
        goto LABEL_28;
      v25 = v9;
      v27.m128i_i64[0] = v10;
      v26 = v11;
      v22 = &v24;
      v23 = 0;
      v24.m128i_i8[0] = 0;
      v28 = 0u;
      if ( v9 != &v27 )
      {
        p_dest = v9;
        v33 = v11;
        dest.m128i_i64[0] = v10;
        v25 = &v27;
        goto LABEL_19;
      }
LABEL_29:
      if ( v11 )
      {
        if ( v11 == 1 )
          dest.m128i_i8[0] = v27.m128i_i8[0];
        else
          memcpy(&dest, &v27, v11);
      }
      v33 = v11;
      dest.m128i_i8[v11] = 0;
LABEL_19:
      v12 = _mm_load_si128(&v28);
      v27.m128i_i8[0] = 0;
      v26 = 0;
      v35 = v12;
      if ( v25 != &v27 )
        j_j___libc_free_0((unsigned __int64)v25);
      if ( v22 != &v24 )
        j_j___libc_free_0((unsigned __int64)v22);
      v13 = *((_BYTE *)v3 + 8);
      v8 = a2[57];
      HIBYTE(v36) = 1;
      LOBYTE(v36) = v13;
      v37 = *((_BYTE *)v3 + 9);
      if ( v8 == a2[58] )
      {
        sub_2F19210(v16, (const __m128i *)v8, (__int64)&v30);
        v6 = p_dest;
      }
      else
      {
        if ( v8 )
        {
          *(__m128i *)v8 = _mm_load_si128(&v30);
          *(_QWORD *)(v8 + 16) = v31;
          *(_QWORD *)(v8 + 24) = v8 + 40;
          if ( p_dest == &dest )
          {
            *(__m128i *)(v8 + 40) = _mm_loadu_si128(&dest);
          }
          else
          {
            *(_QWORD *)(v8 + 24) = p_dest;
            *(_QWORD *)(v8 + 40) = dest.m128i_i64[0];
          }
          v6 = &dest;
          p_dest = &dest;
          *(_QWORD *)(v8 + 32) = v33;
          v7 = _mm_loadu_si128(&v35);
          v33 = 0;
          *(__m128i *)(v8 + 56) = v7;
          dest.m128i_i8[0] = 0;
          *(_WORD *)(v8 + 72) = v36;
          *(_BYTE *)(v8 + 74) = v37;
          v8 = a2[57];
        }
        else
        {
          v6 = p_dest;
        }
        a2[57] = v8 + 80;
      }
      if ( v6 != &dest )
        j_j___libc_free_0((unsigned __int64)v6);
      v29[0] = &unk_49DD210;
      sub_CB5840((__int64)v29);
      if ( v19 != &v21 )
        j_j___libc_free_0((unsigned __int64)v19);
      v3 += 2;
      if ( v17 == v3 )
        return;
      v5 = v18;
    }
    v14 = _mm_load_si128(&v21);
    v21.m128i_i8[0] = 0;
    v11 = n;
    n = 0;
    v25 = &v27;
    v24 = v14;
LABEL_28:
    v15 = _mm_load_si128(&v24);
    v23 = 0;
    v24.m128i_i8[0] = 0;
    v22 = &v24;
    v28 = 0u;
    v27 = v15;
    goto LABEL_29;
  }
}
