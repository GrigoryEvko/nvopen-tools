// Function: sub_3779790
// Address: 0x3779790
//
void __fastcall sub_3779790(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // r14
  __int16 *v9; // rax
  __int64 v10; // rsi
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // r10
  const __m128i *v14; // r15
  __int64 v15; // r11
  unsigned __int64 v16; // rdx
  __int64 v17; // r9
  const __m128i *v18; // rax
  __m128i *v19; // rsi
  int v20; // edi
  _BYTE *v21; // rcx
  unsigned int v22; // esi
  _QWORD *v23; // rdi
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // r8
  __int64 v28; // rax
  const __m128i *v29; // r13
  unsigned __int64 v30; // rdx
  __m128i *v31; // rax
  int v32; // esi
  _QWORD *v33; // rcx
  _QWORD *v34; // rdi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rax
  _QWORD *v37; // rdi
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int128 v42; // [rsp-10h] [rbp-1E0h]
  __int128 v43; // [rsp-10h] [rbp-1E0h]
  __int64 v44; // [rsp+8h] [rbp-1C8h]
  _QWORD *v45; // [rsp+10h] [rbp-1C0h]
  __int64 v46; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v47; // [rsp+20h] [rbp-1B0h]
  const __m128i *v48; // [rsp+20h] [rbp-1B0h]
  __int64 v49; // [rsp+30h] [rbp-1A0h]
  __int64 v50; // [rsp+38h] [rbp-198h]
  __int64 v51; // [rsp+40h] [rbp-190h]
  __int64 v53; // [rsp+70h] [rbp-160h] BYREF
  int v54; // [rsp+78h] [rbp-158h]
  _BYTE *v55; // [rsp+80h] [rbp-150h] BYREF
  __int64 v56; // [rsp+88h] [rbp-148h]
  _BYTE v57[128]; // [rsp+90h] [rbp-140h] BYREF
  _QWORD *v58; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+118h] [rbp-B8h]
  _QWORD v60[22]; // [rsp+120h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v53 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v53, v7, 1);
  v8 = *(_DWORD *)(a2 + 64) >> 1;
  v54 = *(_DWORD *)(a2 + 72);
  if ( (_DWORD)v8 == 1 )
  {
    v39 = *(_QWORD *)(a2 + 40);
    v40 = v53;
    *(_QWORD *)a3 = *(_QWORD *)v39;
    *(_DWORD *)(a3 + 8) = *(_DWORD *)(v39 + 8);
    v41 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)a4 = *(_QWORD *)(v41 + 40);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v41 + 48);
    if ( v40 )
      sub_B91220((__int64)&v53, v40);
  }
  else
  {
    v9 = *(__int16 **)(a2 + 48);
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *v9;
    v12 = *((_QWORD *)v9 + 1);
    LOWORD(v55) = v11;
    v56 = v12;
    sub_33D0340((__int64)&v58, v10, (__int64 *)&v55);
    v13 = 40 * v8;
    v14 = *(const __m128i **)(a2 + 40);
    v56 = 0x800000000LL;
    v51 = v60[0];
    v55 = v57;
    v15 = (__int64)v58;
    v16 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v8) >> 3);
    v50 = v60[1];
    v17 = v59;
    v18 = (const __m128i *)((char *)v14 + 40 * v8);
    if ( (unsigned __int64)(40 * v8) > 0x140 )
    {
      v44 = v59;
      v45 = v58;
      v48 = (const __m128i *)((char *)v14 + v13);
      sub_C8D5F0((__int64)&v55, v57, v16, 0x10u, (__int64)&v55, v59);
      v20 = v56;
      v21 = v55;
      LODWORD(v16) = -858993459 * ((40 * v8) >> 3);
      v18 = v48;
      v13 = 40 * v8;
      v15 = (__int64)v45;
      v17 = v44;
      v19 = (__m128i *)&v55[16 * (unsigned int)v56];
    }
    else
    {
      v19 = (__m128i *)v57;
      v20 = 0;
      v21 = v57;
    }
    if ( v18 != v14 )
    {
      do
      {
        if ( v19 )
          *v19 = _mm_loadu_si128(v14);
        v14 = (const __m128i *)((char *)v14 + 40);
        ++v19;
      }
      while ( v18 != v14 );
      v21 = v55;
      v20 = v56;
    }
    v22 = v16 + v20;
    v23 = *(_QWORD **)(a1 + 8);
    LODWORD(v56) = v22;
    *((_QWORD *)&v42 + 1) = v22;
    *(_QWORD *)&v42 = v21;
    v49 = v13;
    *(_QWORD *)a3 = sub_33FC220(v23, 159, (__int64)&v53, v15, v17, v17, v42);
    v59 = 0x800000000LL;
    *(_DWORD *)(a3 + 8) = v24;
    v25 = *(unsigned int *)(a2 + 64);
    v26 = *(_QWORD *)(a2 + 40);
    v58 = v60;
    v25 *= 40;
    v27 = v26 + v25;
    v28 = v25 - v49;
    v29 = (const __m128i *)(v49 + v26);
    v30 = 0xCCCCCCCCCCCCCCCDLL * (v28 >> 3);
    if ( (unsigned __int64)v28 > 0x140 )
    {
      v46 = v27;
      v47 = 0xCCCCCCCCCCCCCCCDLL * (v28 >> 3);
      sub_C8D5F0((__int64)&v58, v60, v30, 0x10u, v27, (__int64)v60);
      v32 = v59;
      v33 = v58;
      LODWORD(v30) = v47;
      v27 = v46;
      v31 = (__m128i *)&v58[2 * (unsigned int)v59];
    }
    else
    {
      v31 = (__m128i *)v60;
      v32 = 0;
      v33 = v60;
    }
    if ( v29 != (const __m128i *)v27 )
    {
      do
      {
        if ( v31 )
          *v31 = _mm_loadu_si128(v29);
        v29 = (const __m128i *)((char *)v29 + 40);
        ++v31;
      }
      while ( (const __m128i *)v27 != v29 );
      v33 = v58;
      v32 = v59;
    }
    v34 = *(_QWORD **)(a1 + 8);
    v35 = (unsigned int)(v30 + v32);
    LODWORD(v59) = v35;
    *((_QWORD *)&v43 + 1) = v35;
    *(_QWORD *)&v43 = v33;
    v36 = sub_33FC220(v34, 159, (__int64)&v53, v51, v50, (__int64)v60, v43);
    v37 = v58;
    *(_QWORD *)a4 = v36;
    *(_DWORD *)(a4 + 8) = v38;
    if ( v37 != v60 )
      _libc_free((unsigned __int64)v37);
    if ( v55 != v57 )
      _libc_free((unsigned __int64)v55);
    if ( v53 )
      sub_B91220((__int64)&v53, v53);
  }
}
