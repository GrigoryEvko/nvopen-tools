// Function: sub_103E7F0
// Address: 0x103e7f0
//
__int8 __fastcall sub_103E7F0(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 *v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r11
  __int64 v11; // r10
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rdx
  bool v15; // dl
  __int64 v16; // r8
  __m128i v17; // xmm7
  __m128i v18; // xmm6
  __m128i v19; // xmm5
  __m128i v20; // xmm4
  __m128i v21; // xmm2
  __m128i v22; // xmm1
  __m128i v23; // xmm0
  __m128i v24; // xmm3
  __int8 result; // al
  __int64 v26; // rdx
  int v27; // r12d
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  unsigned __int64 v35; // r9
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i *v38; // r12
  unsigned __int64 v39; // rcx
  __m128i *v40; // rdx
  const void *v41; // rsi
  __int8 *v42; // r12
  __int64 v43; // [rsp+0h] [rbp-310h]
  __m128i v44; // [rsp+10h] [rbp-300h] BYREF
  __m128i v45; // [rsp+20h] [rbp-2F0h] BYREF
  __m128i v46; // [rsp+30h] [rbp-2E0h] BYREF
  __m128i v47; // [rsp+40h] [rbp-2D0h] BYREF
  __m128i v48; // [rsp+50h] [rbp-2C0h] BYREF
  __m128i v49; // [rsp+60h] [rbp-2B0h] BYREF
  __m128i v50; // [rsp+70h] [rbp-2A0h] BYREF
  __m128i v51; // [rsp+80h] [rbp-290h] BYREF
  __m128i v52; // [rsp+90h] [rbp-280h] BYREF
  __m128i v53; // [rsp+A0h] [rbp-270h] BYREF
  __m128i v54; // [rsp+B0h] [rbp-260h]
  __m128i v55; // [rsp+C0h] [rbp-250h]
  __m128i v56; // [rsp+D0h] [rbp-240h]
  __m128i v57; // [rsp+E0h] [rbp-230h]
  __m128i v58; // [rsp+F0h] [rbp-220h]
  __m128i v59; // [rsp+100h] [rbp-210h]
  __m128i v60; // [rsp+110h] [rbp-200h]
  __m128i v61; // [rsp+120h] [rbp-1F0h]
  __m128i v62; // [rsp+130h] [rbp-1E0h] BYREF
  __m128i v63; // [rsp+140h] [rbp-1D0h] BYREF
  __m128i v64; // [rsp+150h] [rbp-1C0h] BYREF
  __m128i v65; // [rsp+160h] [rbp-1B0h]
  __m128i v66; // [rsp+170h] [rbp-1A0h]
  __m128i v67; // [rsp+180h] [rbp-190h]
  __m128i v68; // [rsp+190h] [rbp-180h]
  __m128i v69; // [rsp+1A0h] [rbp-170h]
  __m128i v70; // [rsp+1B0h] [rbp-160h]
  __m128i v71; // [rsp+1C0h] [rbp-150h]
  __m128i v72; // [rsp+1D0h] [rbp-140h]
  __m128i v73; // [rsp+1E0h] [rbp-130h]
  __m128i v74; // [rsp+1F0h] [rbp-120h]
  __m128i v75; // [rsp+200h] [rbp-110h]
  __m128i v76; // [rsp+210h] [rbp-100h]
  __m128i v77; // [rsp+220h] [rbp-F0h]
  __m128i v78; // [rsp+230h] [rbp-E0h]
  __m128i v79; // [rsp+240h] [rbp-D0h]

  v6 = a4;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 40) + 72 * v6);
  v9 = *v8;
  v10 = v8[1];
  v44 = 0u;
  v11 = v8[2];
  v12 = v8[3];
  v45 = (__m128i)0xFFFFFFFFFFFFFFFFLL;
  v13 = v8[4];
  v14 = v8[5];
  v46 = 0u;
  v51.m128i_i64[0] = v14;
  v15 = 0;
  v47.m128i_i64[0] = 0;
  v47.m128i_i64[1] = (__int64)a2;
  v48.m128i_i32[0] = 0;
  v48.m128i_i64[1] = v9;
  v49.m128i_i64[0] = v10;
  v49.m128i_i64[1] = v11;
  v50.m128i_i64[0] = v12;
  v50.m128i_i64[1] = v13;
  v51.m128i_i64[1] = (__int64)a2;
  v52.m128i_i64[0] = v7;
  if ( a2 )
    v15 = *a2 == 28;
  v52.m128i_i8[8] = v15;
  sub_103E690((__int64)&v44);
  v17 = _mm_loadu_si128(&v45);
  v18 = _mm_loadu_si128(&v46);
  v19 = _mm_loadu_si128(&v47);
  v20 = _mm_loadu_si128(&v48);
  v21 = _mm_loadu_si128(&v50);
  v53 = _mm_loadu_si128(&v44);
  v22 = _mm_loadu_si128(&v51);
  v23 = _mm_loadu_si128(&v52);
  v54 = v17;
  v24 = _mm_loadu_si128(&v49);
  v55 = v18;
  v56 = v19;
  v57 = v20;
  v58 = v24;
  v59 = v21;
  v60 = v22;
  v61 = v23;
  v62 = v53;
  v63 = v17;
  v64 = v18;
  v65 = v19;
  v66 = v20;
  v67 = v24;
  v68 = v21;
  v69 = v22;
  v70 = v23;
  v71 = v53;
  v72 = v17;
  v73 = v18;
  v74 = v19;
  v75 = v20;
  v76 = v24;
  result = a1 + 40;
  v43 = a1 + 40;
  v77 = v21;
  v78 = v22;
  v79 = v23;
  while ( v65.m128i_i64[1] )
  {
    v26 = *(unsigned int *)(a3 + 8);
    v27 = *(_DWORD *)(a1 + 48);
    if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      result = sub_C8D5F0(a3, (const void *)(a3 + 16), v26 + 1, 4u, v16, v26 + 1);
      v26 = *(unsigned int *)(a3 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a3 + 4 * v26) = v27;
    ++*(_DWORD *)(a3 + 8);
    v28 = *(unsigned int *)(a1 + 48);
    v29 = *(unsigned int *)(a1 + 52);
    v30 = *(_DWORD *)(a1 + 48);
    if ( v28 >= v29 )
    {
      v34 = _mm_loadu_si128((const __m128i *)&v62.m128i_u64[1]);
      v35 = v28 + 1;
      v57.m128i_i32[0] = v6;
      v36 = _mm_loadu_si128((const __m128i *)&v63.m128i_u64[1]);
      v37 = _mm_loadu_si128((const __m128i *)&v64.m128i_u64[1]);
      v57.m128i_i8[4] = 1;
      v56.m128i_i64[0] = v62.m128i_i64[0];
      v38 = &v53;
      v56.m128i_i64[1] = v62.m128i_i64[0];
      v39 = *(_QWORD *)(a1 + 40);
      v53 = v34;
      v54 = v36;
      v55 = v37;
      if ( v29 < v28 + 1 )
      {
        v41 = (const void *)(a1 + 56);
        if ( v39 > (unsigned __int64)&v53 || (unsigned __int64)&v53 >= v39 + 72 * v28 )
        {
          result = sub_C8D5F0(v43, v41, v35, 0x48u, v16, v35);
          v39 = *(_QWORD *)(a1 + 40);
          v28 = *(unsigned int *)(a1 + 48);
        }
        else
        {
          v42 = &v53.m128i_i8[-v39];
          result = sub_C8D5F0(v43, v41, v35, 0x48u, v16, v35);
          v39 = *(_QWORD *)(a1 + 40);
          v28 = *(unsigned int *)(a1 + 48);
          v38 = (__m128i *)&v42[v39];
        }
      }
      v40 = (__m128i *)(v39 + 72 * v28);
      *v40 = _mm_loadu_si128(v38);
      v40[1] = _mm_loadu_si128(v38 + 1);
      v40[2] = _mm_loadu_si128(v38 + 2);
      v40[3] = _mm_loadu_si128(v38 + 3);
      v40[4].m128i_i64[0] = v38[4].m128i_i64[0];
      ++*(_DWORD *)(a1 + 48);
    }
    else
    {
      v31 = *(_QWORD *)(a1 + 40) + 72 * v28;
      if ( v31 )
      {
        v32 = v62.m128i_i64[0];
        *(__m128i *)v31 = _mm_loadu_si128((const __m128i *)&v62.m128i_u64[1]);
        *(__m128i *)(v31 + 16) = _mm_loadu_si128((const __m128i *)&v63.m128i_u64[1]);
        v33 = _mm_loadu_si128((const __m128i *)&v64.m128i_u64[1]);
        *(_QWORD *)(v31 + 48) = v32;
        *(_QWORD *)(v31 + 56) = v32;
        *(_DWORD *)(v31 + 64) = v6;
        *(_BYTE *)(v31 + 68) = 1;
        *(__m128i *)(v31 + 32) = v33;
        v30 = *(_DWORD *)(a1 + 48);
      }
      *(_DWORD *)(a1 + 48) = v30 + 1;
    }
    if ( *(_BYTE *)v65.m128i_i64[1] != 28 )
      break;
    result = v66.m128i_i8[0];
    if ( ++v66.m128i_i32[0] >= (*(_DWORD *)(v65.m128i_i64[1] + 4) & 0x7FFFFFFu) )
      break;
    result = sub_103E690((__int64)&v62);
  }
  return result;
}
