// Function: sub_289E450
// Address: 0x289e450
//
__int64 __fastcall sub_289E450(__int64 a1, __int64 a2, const __m128i *a3, unsigned int **a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r13
  __int32 v9; // edi
  __m128i v10; // xmm1
  __int64 v11; // rax
  unsigned int v12; // esi
  __m128i v13; // xmm0
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 result; // rax
  __int64 v19; // r8
  __int64 v20; // r12
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r10
  __int64 v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rcx
  int v27; // edi
  int v28; // r11d
  __int64 v29; // rdx
  char *v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r9
  __m128i v33; // xmm2
  __int64 *v34; // rcx
  int v35; // edx
  int v36; // edi
  __int64 v37; // rdx
  int v38; // eax
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 **v43; // r11
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rcx
  unsigned __int64 v49; // rdi
  int v50; // eax
  __int64 v51; // rdx
  __int64 v52; // [rsp+10h] [rbp-1D0h]
  int v53; // [rsp+18h] [rbp-1C8h]
  int v54; // [rsp+20h] [rbp-1C0h]
  __int64 v55; // [rsp+20h] [rbp-1C0h]
  __int64 v56; // [rsp+20h] [rbp-1C0h]
  __int64 *v58; // [rsp+38h] [rbp-1A8h] BYREF
  __int64 v59; // [rsp+40h] [rbp-1A0h] BYREF
  int v60; // [rsp+48h] [rbp-198h]
  __int64 v61; // [rsp+50h] [rbp-190h]
  char *v62; // [rsp+58h] [rbp-188h] BYREF
  __int64 v63; // [rsp+60h] [rbp-180h]
  _BYTE v64[128]; // [rsp+68h] [rbp-178h] BYREF
  __m128i v65; // [rsp+E8h] [rbp-F8h] BYREF
  char v66; // [rsp+F8h] [rbp-E8h]
  __int64 v67; // [rsp+100h] [rbp-E0h]
  char *v68; // [rsp+108h] [rbp-D8h] BYREF
  __int64 v69; // [rsp+110h] [rbp-D0h]
  _BYTE v70[128]; // [rsp+118h] [rbp-C8h] BYREF
  __m128i v71; // [rsp+198h] [rbp-48h] BYREF
  unsigned __int8 v72; // [rsp+1A8h] [rbp-38h]

  v6 = a3;
  v9 = a3->m128i_i32[2];
  v61 = a2;
  v62 = v64;
  v63 = 0x1000000000LL;
  if ( v9 )
  {
    sub_2894AD0((__int64)&v62, (__int64)a3, (__int64)a3, (__int64)a4, (__int64)&v62, a6);
    v11 = v61;
    v33 = _mm_loadu_si128(v6 + 9);
    a3 = (const __m128i *)v6[10].m128i_u8[0];
    v68 = v70;
    v67 = v61;
    v66 = (char)a3;
    v69 = 0x1000000000LL;
    v65 = v33;
    if ( (_DWORD)v63 )
    {
      sub_2894810((__int64)&v68, &v62, (__int64)a3, v31, (__int64)&v62, v32);
      LOBYTE(a3) = v66;
      v11 = v67;
    }
  }
  else
  {
    v10 = _mm_loadu_si128(a3 + 9);
    LOBYTE(a3) = a3[10].m128i_i8[0];
    v67 = a2;
    v68 = v70;
    v11 = a2;
    v66 = (char)a3;
    v69 = 0x1000000000LL;
    v65 = v10;
  }
  v12 = *(_DWORD *)(a1 + 264);
  v13 = _mm_loadu_si128(&v65);
  v72 = (unsigned __int8)a3;
  v59 = v11;
  v60 = 0;
  v71 = v13;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 240);
    v58 = 0;
LABEL_54:
    v12 *= 2;
LABEL_55:
    sub_D39D40(a1 + 240, v12);
    sub_22B1A50(a1 + 240, &v59, &v58);
    v11 = v59;
    v34 = v58;
    v36 = *(_DWORD *)(a1 + 256) + 1;
    goto LABEL_45;
  }
  v14 = *(_QWORD *)(a1 + 248);
  v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = (__int64 *)(v14 + 16 * v15);
  v17 = *v16;
  if ( v11 == *v16 )
    goto LABEL_5;
  v54 = 1;
  v34 = 0;
  while ( v17 != -4096 )
  {
    if ( v34 || v17 != -8192 )
      v16 = v34;
    v15 = (v12 - 1) & (v54 + (_DWORD)v15);
    v17 = *(_QWORD *)(v14 + 16LL * (unsigned int)v15);
    if ( v11 == v17 )
      goto LABEL_5;
    ++v54;
    v34 = v16;
    v16 = (__int64 *)(v14 + 16LL * (unsigned int)v15);
  }
  if ( !v34 )
    v34 = v16;
  v35 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  v36 = v35 + 1;
  v58 = v34;
  if ( 4 * (v35 + 1) >= 3 * v12 )
    goto LABEL_54;
  v14 = v12 >> 3;
  if ( v12 - *(_DWORD *)(a1 + 260) - v36 <= (unsigned int)v14 )
    goto LABEL_55;
LABEL_45:
  *(_DWORD *)(a1 + 256) = v36;
  if ( *v34 != -4096 )
    --*(_DWORD *)(a1 + 260);
  *v34 = v11;
  *((_DWORD *)v34 + 2) = v60;
  *((_DWORD *)v34 + 2) = *(_DWORD *)(a1 + 280);
  v37 = *(unsigned int *)(a1 + 280);
  v38 = v37;
  if ( *(_DWORD *)(a1 + 284) <= (unsigned int)v37 )
  {
    v40 = sub_C8D7D0(a1 + 272, a1 + 288, 0, 0xB0u, (unsigned __int64 *)&v59, v15);
    v43 = (__int64 **)(a1 + 272);
    v44 = v40;
    v45 = *(unsigned int *)(a1 + 280);
    v46 = 5 * v45;
    v47 = v44 + 176 * v45;
    if ( v47 )
    {
      v48 = v67;
      *(_QWORD *)(v47 + 16) = 0x1000000000LL;
      *(_QWORD *)v47 = v48;
      *(_QWORD *)(v47 + 8) = v47 + 24;
      if ( (_DWORD)v69 )
      {
        v52 = v44;
        v56 = v47;
        sub_2894810(v47 + 8, &v68, v44, (unsigned int)v69, v41, v42);
        v44 = v52;
        v43 = (__int64 **)(a1 + 272);
        v47 = v56;
      }
      *(__m128i *)(v47 + 152) = _mm_loadu_si128(&v71);
      v46 = v72;
      *(_BYTE *)(v47 + 168) = v72;
    }
    v55 = v44;
    sub_2894F90(v43, v44, v44, v46, v41, v42);
    v49 = *(_QWORD *)(a1 + 272);
    v50 = v59;
    v51 = v55;
    if ( a1 + 288 != v49 )
    {
      v53 = v59;
      _libc_free(v49);
      v50 = v53;
      v51 = v55;
    }
    ++*(_DWORD *)(a1 + 280);
    *(_QWORD *)(a1 + 272) = v51;
    *(_DWORD *)(a1 + 284) = v50;
  }
  else
  {
    v39 = *(_QWORD *)(a1 + 272) + 176 * v37;
    if ( v39 )
    {
      *(_QWORD *)v39 = v67;
      *(_QWORD *)(v39 + 8) = v39 + 24;
      *(_QWORD *)(v39 + 16) = 0x1000000000LL;
      if ( (_DWORD)v69 )
        sub_2894810(v39 + 8, &v68, v37, 5 * v37, v14, v15);
      *(__m128i *)(v39 + 152) = _mm_loadu_si128(&v71);
      *(_BYTE *)(v39 + 168) = v72;
      v38 = *(_DWORD *)(a1 + 280);
    }
    *(_DWORD *)(a1 + 280) = v38 + 1;
  }
LABEL_5:
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  result = *(unsigned int *)(a1 + 104);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
  {
    sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), result + 1, 8u, v14, v15);
    result = *(unsigned int *)(a1 + 104);
  }
  v19 = 0;
  *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 104);
  v20 = *(_QWORD *)(a2 + 16);
  if ( v20 )
  {
    while ( 1 )
    {
      result = *(unsigned int *)(a1 + 88);
      v24 = v20;
      v25 = *(_QWORD *)(a1 + 72);
      v20 = *(_QWORD *)(v20 + 8);
      v26 = *(_QWORD *)(v24 + 24);
      if ( !(_DWORD)result )
        goto LABEL_17;
      v21 = (result - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v22 = (__int64 *)(v25 + 24LL * v21);
      v23 = *v22;
      if ( v26 == *v22 )
      {
LABEL_14:
        result = v25 + 24 * result;
        if ( v22 == (__int64 *)result )
          goto LABEL_17;
LABEL_15:
        if ( !v20 )
          return result;
      }
      else
      {
        v27 = 1;
        while ( v23 != -4096 )
        {
          v28 = v27 + 1;
          v21 = (result - 1) & (v27 + v21);
          v22 = (__int64 *)(v25 + 24LL * v21);
          v23 = *v22;
          if ( v26 == *v22 )
            goto LABEL_14;
          v27 = v28;
        }
LABEL_17:
        if ( v19 )
        {
          if ( !*(_QWORD *)v24 || (**(_QWORD **)(v24 + 16) = v20, (result = v20) == 0) )
          {
            *(_QWORD *)v24 = v19;
            goto LABEL_22;
          }
LABEL_20:
          *(_QWORD *)(result + 16) = *(_QWORD *)(v24 + 16);
          goto LABEL_21;
        }
        v29 = v6->m128i_u32[2];
        v30 = (char *)v6->m128i_i64[0];
        if ( v29 == 1 )
        {
          v19 = *(_QWORD *)v30;
        }
        else
        {
          result = sub_9B9840(a4, v30, v29);
          v19 = result;
        }
        if ( *(_QWORD *)v24 )
        {
          result = *(_QWORD *)(v24 + 8);
          **(_QWORD **)(v24 + 16) = result;
          if ( result )
            goto LABEL_20;
        }
LABEL_21:
        *(_QWORD *)v24 = v19;
        if ( !v19 )
          goto LABEL_15;
LABEL_22:
        result = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(v24 + 8) = result;
        if ( result )
          *(_QWORD *)(result + 16) = v24 + 8;
        *(_QWORD *)(v24 + 16) = v19 + 16;
        *(_QWORD *)(v19 + 16) = v24;
        if ( !v20 )
          return result;
      }
    }
  }
  return result;
}
