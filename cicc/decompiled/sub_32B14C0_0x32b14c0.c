// Function: sub_32B14C0
// Address: 0x32b14c0
//
__int64 __fastcall sub_32B14C0(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rcx
  __int64 v9; // r8
  __m128i *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __m128i *v17; // r14
  char v18; // dl
  bool v19; // r8
  __m128i si128; // xmm0
  __int64 v21; // rdx
  __int64 v22; // rax
  const __m128i *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  bool v26; // r10
  __m128i *v27; // rax
  _QWORD *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // r12
  __m128i *v32; // r14
  _BOOL4 v33; // r8d
  char v34; // di
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  char v38; // [rsp+Ch] [rbp-54h]
  _QWORD *v39; // [rsp+10h] [rbp-50h]
  char v40; // [rsp+18h] [rbp-48h]
  _QWORD *v41; // [rsp+18h] [rbp-48h]
  __m128i v42[4]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 184) )
  {
    v14 = sub_3055070(a2 + 144, (unsigned __int64 *)a3);
    v16 = v15;
    v17 = (__m128i *)v14;
    v18 = 0;
    if ( v16 )
    {
      v19 = 1;
      if ( !v14 && v16 != a2 + 152 )
      {
        v36 = *(_QWORD *)(v16 + 32);
        v19 = a3->m128i_i64[0] < v36 || a3->m128i_i64[0] == v36 && a3->m128i_i32[2] < *(_DWORD *)(v16 + 40);
      }
      v40 = v19;
      v42[0].m128i_i64[0] = a2 + 152;
      v17 = (__m128i *)sub_22077B0(0x30u);
      v17[2] = _mm_loadu_si128(a3);
      sub_220F040(v40, (__int64)v17, (_QWORD *)v16, (_QWORD *)(a2 + 152));
      ++*(_QWORD *)(a2 + 184);
      v18 = 1;
    }
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v17;
    *(_BYTE *)(a1 + 16) = v18;
  }
  else
  {
    v8 = *(unsigned int *)(a2 + 8);
    v9 = *(_QWORD *)a2;
    v10 = (__m128i *)(*(_QWORD *)a2 + 16 * v8);
    if ( *(__m128i **)a2 == v10 )
    {
      if ( v8 <= 7 )
      {
LABEL_15:
        si128 = _mm_loadu_si128(a3);
        if ( v8 + 1 > *(unsigned int *)(a2 + 12) )
        {
          v42[0] = si128;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v8 + 1, 0x10u, v9, a6);
          si128 = _mm_load_si128(v42);
          v10 = (__m128i *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
        }
        *v10 = si128;
        v21 = *(_QWORD *)a2;
        v22 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v22;
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v21 + 16 * v22 - 16;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v41 = (_QWORD *)(a2 + 144);
    }
    else
    {
      v11 = a3->m128i_i64[0];
      v12 = *(_QWORD *)a2;
      while ( *(_QWORD *)v12 != v11 || *(_DWORD *)(v12 + 8) != a3->m128i_i32[2] )
      {
        v12 += 16;
        if ( v10 == (__m128i *)v12 )
          goto LABEL_14;
      }
      if ( v10 != (__m128i *)v12 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v12;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_14:
      if ( v8 <= 7 )
        goto LABEL_15;
      v23 = *(const __m128i **)a2;
      v41 = (_QWORD *)(a2 + 144);
      v42[0].m128i_i64[0] = a2 + 152;
      do
      {
        v24 = sub_3055200(v41, v42[0].m128i_i64[0], (__int64)v23);
        if ( v25 )
        {
          v26 = 1;
          if ( !v24 && v42[0].m128i_i64[0] != v25 )
          {
            v35 = *(_QWORD *)(v25 + 32);
            if ( v23->m128i_i64[0] >= v35 && (v23->m128i_i64[0] != v35 || v23->m128i_i32[2] >= *(_DWORD *)(v25 + 40)) )
              v26 = 0;
          }
          v38 = v26;
          v39 = (_QWORD *)v25;
          v27 = (__m128i *)sub_22077B0(0x30u);
          v28 = (_QWORD *)v42[0].m128i_i64[0];
          v27[2] = _mm_loadu_si128(v23);
          sub_220F040(v38, (__int64)v27, v39, v28);
          ++*(_QWORD *)(a2 + 184);
        }
        ++v23;
      }
      while ( v10 != v23 );
    }
    *(_DWORD *)(a2 + 8) = 0;
    v29 = sub_3055070((__int64)v41, (unsigned __int64 *)a3);
    v31 = (_QWORD *)v30;
    v32 = (__m128i *)v29;
    if ( v30 )
    {
      v33 = 1;
      if ( !v29 && v30 != a2 + 152 )
      {
        v37 = *(_QWORD *)(v30 + 32);
        v33 = a3->m128i_i64[0] < v37 || a3->m128i_i64[0] == v37 && a3->m128i_i32[2] < *(_DWORD *)(v30 + 40);
      }
      v42[0].m128i_i32[0] = v33;
      v32 = (__m128i *)sub_22077B0(0x30u);
      v34 = v42[0].m128i_i8[0];
      v32[2] = _mm_loadu_si128(a3);
      sub_220F040(v34, (__int64)v32, v31, (_QWORD *)(a2 + 152));
      ++*(_QWORD *)(a2 + 184);
    }
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v32;
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
