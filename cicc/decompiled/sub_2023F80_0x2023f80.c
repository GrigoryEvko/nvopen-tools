// Function: sub_2023F80
// Address: 0x2023f80
//
void __fastcall sub_2023F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128 a5, __m128 a6, __m128i a7)
{
  __int64 v10; // rsi
  __int64 v11; // r14
  char *v12; // rax
  __int64 v13; // rsi
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // r11
  const __m128i *v17; // r15
  __int64 v18; // r9
  unsigned __int64 v19; // rdx
  const void **v20; // r10
  const __m128i *v21; // rax
  __m128 *v22; // rsi
  int v23; // edi
  _BYTE *v24; // rcx
  unsigned int v25; // esi
  __int64 *v26; // rdi
  int v27; // edx
  __int64 v28; // rdx
  __int64 v29; // rax
  const __m128i *v30; // r8
  __int64 v31; // rdx
  const __m128i *v32; // rax
  unsigned __int64 v33; // r9
  __m128 *v34; // rdx
  int v35; // esi
  _QWORD *v36; // rcx
  __int64 *v37; // rdi
  __int64 *v38; // rax
  _QWORD *v39; // rdi
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int128 v44; // [rsp-10h] [rbp-1E0h]
  __int128 v45; // [rsp-10h] [rbp-1E0h]
  const void **v46; // [rsp+8h] [rbp-1C8h]
  _QWORD *v47; // [rsp+10h] [rbp-1C0h]
  const __m128i *v48; // [rsp+18h] [rbp-1B8h]
  const __m128i *v49; // [rsp+20h] [rbp-1B0h]
  const __m128i *v50; // [rsp+20h] [rbp-1B0h]
  __int64 v51; // [rsp+30h] [rbp-1A0h]
  int v52; // [rsp+30h] [rbp-1A0h]
  const void **v53; // [rsp+38h] [rbp-198h]
  __int64 v54; // [rsp+40h] [rbp-190h]
  __int64 v56; // [rsp+70h] [rbp-160h] BYREF
  int v57; // [rsp+78h] [rbp-158h]
  _BYTE *v58; // [rsp+80h] [rbp-150h] BYREF
  __int64 v59; // [rsp+88h] [rbp-148h]
  _BYTE v60[128]; // [rsp+90h] [rbp-140h] BYREF
  _QWORD *v61; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+118h] [rbp-B8h]
  _QWORD v63[22]; // [rsp+120h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v56 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v56, v10, 2);
  v11 = *(_DWORD *)(a2 + 56) >> 1;
  v57 = *(_DWORD *)(a2 + 64);
  if ( (_DWORD)v11 == 1 )
  {
    v41 = *(_QWORD *)(a2 + 32);
    v42 = v56;
    *(_QWORD *)a3 = *(_QWORD *)v41;
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v41 + 8);
    v43 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)a4 = *(_QWORD *)(v43 + 40);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v43 + 48);
    if ( v42 )
      sub_161E7C0((__int64)&v56, v42);
  }
  else
  {
    v12 = *(char **)(a2 + 40);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *v12;
    v15 = *((_QWORD *)v12 + 1);
    LOBYTE(v58) = v14;
    v59 = v15;
    sub_1D19A30((__int64)&v61, v13, &v58);
    v16 = 40 * v11;
    v17 = *(const __m128i **)(a2 + 32);
    v59 = 0x800000000LL;
    v54 = v63[0];
    v58 = v60;
    v18 = (__int64)v61;
    v19 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v11) >> 3);
    v53 = (const void **)v63[1];
    v20 = (const void **)v62;
    v21 = (const __m128i *)((char *)v17 + 40 * v11);
    if ( (unsigned __int64)(40 * v11) > 0x140 )
    {
      v46 = (const void **)v62;
      v47 = v61;
      v50 = (const __m128i *)((char *)v17 + v16);
      sub_16CD150((__int64)&v58, v60, v19, 16, (int)&v58, (int)v61);
      v23 = v59;
      v24 = v58;
      LODWORD(v19) = -858993459 * ((40 * v11) >> 3);
      v21 = v50;
      v16 = 40 * v11;
      v18 = (__int64)v47;
      v20 = v46;
      v22 = (__m128 *)&v58[16 * (unsigned int)v59];
    }
    else
    {
      v22 = (__m128 *)v60;
      v23 = 0;
      v24 = v60;
    }
    if ( v21 != v17 )
    {
      do
      {
        if ( v22 )
        {
          a5 = (__m128)_mm_loadu_si128(v17);
          *v22 = a5;
        }
        v17 = (const __m128i *)((char *)v17 + 40);
        ++v22;
      }
      while ( v21 != v17 );
      v24 = v58;
      v23 = v59;
    }
    v25 = v19 + v23;
    v26 = *(__int64 **)(a1 + 8);
    *((_QWORD *)&v44 + 1) = v25;
    *(_QWORD *)&v44 = v24;
    LODWORD(v59) = v25;
    v51 = v16;
    *(_QWORD *)a3 = sub_1D359D0(
                      v26,
                      107,
                      (__int64)&v56,
                      v18,
                      v20,
                      0,
                      *(double *)a5.m128_u64,
                      *(double *)a6.m128_u64,
                      a7,
                      v44);
    v62 = 0x800000000LL;
    *(_DWORD *)(a3 + 8) = v27;
    v28 = *(unsigned int *)(a2 + 56);
    v29 = *(_QWORD *)(a2 + 32);
    v61 = v63;
    v28 *= 40;
    v30 = (const __m128i *)(v29 + v28);
    v31 = v28 - v51;
    v32 = (const __m128i *)(v51 + v29);
    v33 = 0xCCCCCCCCCCCCCCCDLL * (v31 >> 3);
    if ( (unsigned __int64)v31 > 0x140 )
    {
      v48 = v32;
      v49 = v30;
      v52 = -858993459 * (v31 >> 3);
      sub_16CD150((__int64)&v61, v63, 0xCCCCCCCCCCCCCCCDLL * (v31 >> 3), 16, (int)v30, v33);
      v35 = v62;
      v36 = v61;
      LODWORD(v33) = v52;
      v30 = v49;
      v32 = v48;
      v34 = (__m128 *)&v61[2 * (unsigned int)v62];
    }
    else
    {
      v34 = (__m128 *)v63;
      v35 = 0;
      v36 = v63;
    }
    if ( v32 != v30 )
    {
      do
      {
        if ( v34 )
        {
          a6 = (__m128)_mm_loadu_si128(v32);
          *v34 = a6;
        }
        v32 = (const __m128i *)((char *)v32 + 40);
        ++v34;
      }
      while ( v30 != v32 );
      v36 = v61;
      v35 = v62;
    }
    v37 = *(__int64 **)(a1 + 8);
    LODWORD(v62) = v35 + v33;
    *((_QWORD *)&v45 + 1) = (unsigned int)(v35 + v33);
    *(_QWORD *)&v45 = v36;
    v38 = sub_1D359D0(v37, 107, (__int64)&v56, v54, v53, 0, *(double *)a5.m128_u64, *(double *)a6.m128_u64, a7, v45);
    v39 = v61;
    *(_QWORD *)a4 = v38;
    *(_DWORD *)(a4 + 8) = v40;
    if ( v39 != v63 )
      _libc_free((unsigned __int64)v39);
    if ( v58 != v60 )
      _libc_free((unsigned __int64)v58);
    if ( v56 )
      sub_161E7C0((__int64)&v56, v56);
  }
}
