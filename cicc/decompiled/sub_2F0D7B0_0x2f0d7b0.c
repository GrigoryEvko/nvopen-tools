// Function: sub_2F0D7B0
// Address: 0x2f0d7b0
//
void __fastcall sub_2F0D7B0(__int64 a1, __int64 a2)
{
  __m128i *p_src; // rbx
  __int64 v3; // r14
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __m128i v9; // xmm1
  __int64 v10; // rax
  __m128i v11; // xmm2
  bool v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __m128i *v15; // rdi
  int v16; // eax
  size_t v17; // rdx
  __m128i v18; // xmm3
  __int64 v19; // rax
  __int64 v20; // r14
  __m128i *v21; // r12
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rax
  _BYTE *v25; // rax
  __m128i *v26; // r14
  size_t v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __m128i *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // rsi
  int v34; // eax
  unsigned __int64 v35; // rdi
  size_t v36; // rdx
  __int64 v37; // rdi
  __m128i *v38; // [rsp+10h] [rbp-E0h]
  __m128i *v39; // [rsp+18h] [rbp-D8h]
  __int64 v41; // [rsp+40h] [rbp-B0h]
  _QWORD *v42; // [rsp+48h] [rbp-A8h] BYREF
  _QWORD v43[2]; // [rsp+58h] [rbp-98h] BYREF
  __m128i v44; // [rsp+68h] [rbp-88h]
  int v45; // [rsp+78h] [rbp-78h]
  __int64 v46; // [rsp+80h] [rbp-70h]
  unsigned __int64 v47; // [rsp+88h] [rbp-68h] BYREF
  size_t n; // [rsp+90h] [rbp-60h]
  __m128i src; // [rsp+98h] [rbp-58h] BYREF
  __m128i v50; // [rsp+A8h] [rbp-48h] BYREF
  int v51; // [rsp+B8h] [rbp-38h]

  if ( a1 != a2 && a1 + 64 != a2 )
  {
    p_src = &src;
    v3 = a1 + 64;
    v38 = (__m128i *)(a1 + 24);
    while ( 1 )
    {
      v5 = *(_BYTE **)(a1 + 8);
      v6 = *(_QWORD *)(a1 + 16);
      v47 = (unsigned __int64)p_src;
      v46 = *(_QWORD *)a1;
      sub_2F07250((__int64 *)&v47, v5, (__int64)&v5[v6]);
      v7 = *(_BYTE **)(v3 + 8);
      v8 = *(_QWORD *)(v3 + 16);
      v9 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      v51 = *(_DWORD *)(a1 + 56);
      v10 = *(_QWORD *)v3;
      v50 = v9;
      v41 = v10;
      v42 = v43;
      sub_2F07250((__int64 *)&v42, v7, (__int64)&v7[v8]);
      v11 = _mm_loadu_si128((const __m128i *)(v3 + 40));
      v45 = *(_DWORD *)(v3 + 56);
      v44 = v11;
      v12 = (unsigned int)v41 < (unsigned int)v46;
      if ( (_DWORD)v41 == (_DWORD)v46 )
        v12 = HIDWORD(v41) < HIDWORD(v46);
      if ( v42 != v43 )
        j_j___libc_free_0((unsigned __int64)v42);
      if ( (__m128i *)v47 != p_src )
        j_j___libc_free_0(v47);
      v13 = v3 + 64;
      if ( !v12 )
      {
        sub_2F0D3A0((__int64 *)v3);
        if ( a2 == v13 )
          return;
        goto LABEL_35;
      }
      v14 = *(_QWORD *)v3;
      v15 = (__m128i *)(v3 + 24);
      v47 = (unsigned __int64)p_src;
      v46 = v14;
      if ( v3 + 24 == *(_QWORD *)(v3 + 8) )
      {
        src = _mm_loadu_si128((const __m128i *)(v3 + 24));
      }
      else
      {
        v47 = *(_QWORD *)(v3 + 8);
        src.m128i_i64[0] = *(_QWORD *)(v3 + 24);
      }
      v16 = *(_DWORD *)(v3 + 56);
      v17 = *(_QWORD *)(v3 + 16);
      *(_QWORD *)(v3 + 8) = v15;
      v18 = _mm_loadu_si128((const __m128i *)(v3 + 40));
      *(_QWORD *)(v3 + 16) = 0;
      v51 = v16;
      v19 = v3 - a1;
      n = v17;
      *(_BYTE *)(v3 + 24) = 0;
      v50 = v18;
      v20 = (v3 - a1) >> 6;
      if ( v19 > 0 )
      {
        v39 = p_src;
        v21 = (__m128i *)(v13 - 104);
        v22 = v20;
        while ( 1 )
        {
          v26 = (__m128i *)v21[-1].m128i_i64[0];
          v21[2].m128i_i64[1] = v21[-2].m128i_i64[1];
          if ( v21 == v26 )
          {
            v27 = v21[-1].m128i_u64[1];
            if ( v27 )
            {
              if ( v27 == 1 )
                v15->m128i_i8[0] = v21->m128i_i8[0];
              else
                memcpy(v15, v21, v27);
            }
            v28 = v26[-1].m128i_i64[1];
            v29 = v26[3].m128i_i64[0];
            v26[3].m128i_i64[1] = v28;
            *(_BYTE *)(v29 + v28) = 0;
          }
          else
          {
            if ( v15 == &v21[4] )
            {
              v30 = v21[-1].m128i_i64[1];
              v21[3].m128i_i64[0] = (__int64)v26;
              v21[3].m128i_i64[1] = v30;
              v21[4].m128i_i64[0] = v21->m128i_i64[0];
            }
            else
            {
              v23 = v21[-1].m128i_i64[1];
              v24 = v21[4].m128i_i64[0];
              v21[3].m128i_i64[0] = (__int64)v26;
              v21[3].m128i_i64[1] = v23;
              v21[4].m128i_i64[0] = v21->m128i_i64[0];
              if ( v15 )
              {
                v21[-1].m128i_i64[0] = (__int64)v15;
                v21->m128i_i64[0] = v24;
                goto LABEL_18;
              }
            }
            v21[-1].m128i_i64[0] = (__int64)v21;
          }
LABEL_18:
          v25 = (_BYTE *)v21[-1].m128i_i64[0];
          v21 -= 4;
          v21[3].m128i_i64[1] = 0;
          *v25 = 0;
          LODWORD(v25) = v21[6].m128i_i32[0];
          v21[9] = _mm_loadu_si128(v21 + 5);
          v21[10].m128i_i32[0] = (int)v25;
          if ( !--v22 )
          {
            p_src = v39;
            v17 = n;
            break;
          }
          v15 = (__m128i *)v21[3].m128i_i64[0];
        }
      }
      v31 = *(__m128i **)(a1 + 8);
      *(_QWORD *)a1 = v46;
      if ( (__m128i *)v47 == p_src )
      {
        if ( !v17 )
          goto LABEL_43;
        if ( v17 != 1 )
        {
          memcpy(v31, p_src, v17);
          v17 = n;
          v31 = *(__m128i **)(a1 + 8);
LABEL_43:
          *(_QWORD *)(a1 + 16) = v17;
          v31->m128i_i8[v17] = 0;
          v31 = (__m128i *)v47;
          goto LABEL_32;
        }
        v31->m128i_i8[0] = src.m128i_i8[0];
        v36 = n;
        v37 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = n;
        *(_BYTE *)(v37 + v36) = 0;
        v31 = (__m128i *)v47;
      }
      else
      {
        v32 = src.m128i_i64[0];
        if ( v31 == v38 )
        {
          *(_QWORD *)(a1 + 8) = v47;
          *(_QWORD *)(a1 + 16) = v17;
          *(_QWORD *)(a1 + 24) = v32;
        }
        else
        {
          v33 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 8) = v47;
          *(_QWORD *)(a1 + 16) = v17;
          *(_QWORD *)(a1 + 24) = v32;
          if ( v31 )
          {
            v47 = (unsigned __int64)v31;
            src.m128i_i64[0] = v33;
            goto LABEL_32;
          }
        }
        v47 = (unsigned __int64)p_src;
        v31 = &src;
        p_src = &src;
      }
LABEL_32:
      n = 0;
      v31->m128i_i8[0] = 0;
      v34 = v51;
      v35 = v47;
      *(__m128i *)(a1 + 40) = _mm_loadu_si128(&v50);
      *(_DWORD *)(a1 + 56) = v34;
      if ( (__m128i *)v35 != p_src )
        j_j___libc_free_0(v35);
      if ( a2 == v13 )
        return;
LABEL_35:
      v3 = v13;
    }
  }
}
