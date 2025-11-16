// Function: sub_266C480
// Address: 0x266c480
//
__int64 __fastcall sub_266C480(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // rdi
  int v9; // r11d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  int v13; // eax
  unsigned int v14; // r13d
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int64 v20; // rdx
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __int32 v25; // edx
  __int64 v26; // rdi
  __int64 v27; // rax
  __m128i *v29; // [rsp-100h] [rbp-100h]
  unsigned int v30; // [rsp-ECh] [rbp-ECh]
  __m128i v31; // [rsp-E8h] [rbp-E8h] BYREF
  __m128i v32; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v33; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v34; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v35; // [rsp-A8h] [rbp-A8h] BYREF
  int v36; // [rsp-98h] [rbp-98h]
  __m128i v37; // [rsp-88h] [rbp-88h] BYREF
  __m128i v38; // [rsp-78h] [rbp-78h] BYREF
  __m128i v39; // [rsp-68h] [rbp-68h] BYREF
  __m128i v40; // [rsp-58h] [rbp-58h] BYREF
  __m128i v41; // [rsp-48h] [rbp-48h] BYREF
  __int32 v42; // [rsp-38h] [rbp-38h]

  result = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    v7 = *(unsigned int *)(a2 + 88);
    v8 = *(_QWORD *)(a2 + 72);
    if ( !(_DWORD)v7 )
      goto LABEL_8;
    v9 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_502F110 >> 9) ^ ((unsigned int)&unk_502F110 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)))); ; i = (v7 - 1) & v12 )
    {
      v11 = v8 + 24LL * i;
      if ( *(_UNKNOWN **)v11 == &unk_502F110 && a4 == *(_QWORD *)(v11 + 8) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
        goto LABEL_8;
      v12 = v9 + i;
      ++v9;
    }
    if ( v11 != v8 + 24 * v7 && (v27 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL)) != 0 )
    {
      return *(_QWORD *)(v27 + 24);
    }
    else
    {
LABEL_8:
      v13 = *(_DWORD *)(a1 + 88);
      v31 = _mm_loadu_si128((const __m128i *)(a1 + 8));
      v14 = *(_DWORD *)(a1 + 96);
      v32 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      v36 = v13;
      v33 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      v34 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      v35 = _mm_loadu_si128((const __m128i *)(a1 + 72));
      result = sub_22077B0(0xA8u);
      if ( result )
      {
        v15 = _mm_loadu_si128(&v31);
        v16 = _mm_loadu_si128(&v32);
        v17 = _mm_loadu_si128(&v33);
        v18 = _mm_loadu_si128(&v34);
        v19 = _mm_loadu_si128(&v35);
        v42 = v36;
        v20 = a3;
        LOBYTE(v30) = 1;
        v29 = (__m128i *)result;
        v37 = v15;
        v38 = v16;
        v39 = v17;
        v40 = v18;
        v41 = v19;
        sub_30CBEF0(result, a4, v20, v14 | 0x300000000LL, v30);
        v21 = _mm_loadu_si128(&v38);
        v22 = _mm_loadu_si128(&v39);
        v23 = _mm_loadu_si128(&v40);
        v24 = _mm_loadu_si128(&v41);
        v29[5] = _mm_loadu_si128(&v37);
        v29->m128i_i64[0] = (__int64)&unk_4A32558;
        v25 = v42;
        v29[6] = v21;
        v29[10].m128i_i32[0] = v25;
        v29[7] = v22;
        v29[8] = v23;
        v29[9] = v24;
        sub_30CA8B0(v29);
        result = (__int64)v29;
      }
      v26 = *(_QWORD *)a1;
      *(_QWORD *)a1 = result;
      if ( v26 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
        return *(_QWORD *)a1;
      }
    }
  }
  return result;
}
