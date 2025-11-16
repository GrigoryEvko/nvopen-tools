// Function: sub_21D6D20
// Address: 0x21d6d20
//
__int64 *__fastcall sub_21D6D20(__m128i a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r15
  __int64 v13; // r14
  __int128 v14; // rax
  __int64 *v15; // rax
  int v16; // edx
  __int128 v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // r8
  int v20; // edx
  unsigned __int64 v21; // r15
  int v22; // r9d
  _BYTE *v23; // rax
  __int64 v24; // r15
  _BYTE *i; // rdx
  unsigned __int64 v26; // r9
  const void *v27; // r10
  size_t v28; // r11
  __int64 v29; // rdx
  __int64 v30; // r9
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *result; // rax
  unsigned __int8 *v36; // rdi
  __int128 v37; // [rsp-10h] [rbp-1D0h]
  __int64 *v38; // [rsp+0h] [rbp-1C0h]
  __int64 v39; // [rsp+8h] [rbp-1B8h]
  __int64 *v40; // [rsp+10h] [rbp-1B0h]
  const void *v41; // [rsp+10h] [rbp-1B0h]
  __int64 *v42; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v43; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v44; // [rsp+18h] [rbp-1A8h]
  int v45; // [rsp+20h] [rbp-1A0h]
  int v46; // [rsp+28h] [rbp-198h]
  __int64 *v47; // [rsp+30h] [rbp-190h]
  __int64 *v48; // [rsp+30h] [rbp-190h]
  __int64 *v49; // [rsp+30h] [rbp-190h]
  __int64 *v50; // [rsp+30h] [rbp-190h]
  __int64 v51; // [rsp+60h] [rbp-160h] BYREF
  int v52; // [rsp+68h] [rbp-158h]
  _BYTE *v53; // [rsp+70h] [rbp-150h] BYREF
  __int64 v54; // [rsp+78h] [rbp-148h]
  _BYTE v55[128]; // [rsp+80h] [rbp-140h] BYREF
  unsigned __int8 *v56; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+108h] [rbp-B8h]
  _BYTE dest[176]; // [rsp+110h] [rbp-B0h] BYREF

  v9 = *(_QWORD *)(a5 + 72);
  v51 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v51, v9, 2);
  v52 = *(_DWORD *)(a5 + 64);
  v10 = sub_1D309E0(
          a7,
          158,
          (__int64)&v51,
          50,
          0,
          0,
          *(double *)a1.m128i_i64,
          a2,
          *(double *)a3.m128i_i64,
          *(_OWORD *)(*(_QWORD *)(a5 + 32) + 80LL));
  v12 = v11;
  v13 = v10;
  *(_QWORD *)&v14 = sub_1D38E70((__int64)a7, 0, (__int64)&v51, 0, a1, a2, a3);
  v15 = sub_1D332F0(a7, 106, (__int64)&v51, 6, 0, 0, *(double *)a1.m128i_i64, a2, a3, v13, v12, v14);
  v46 = v16;
  v47 = v15;
  *(_QWORD *)&v17 = sub_1D38E70((__int64)a7, 1, (__int64)&v51, 0, a1, a2, a3);
  v18 = sub_1D332F0(a7, 106, (__int64)&v51, 6, 0, 0, *(double *)a1.m128i_i64, a2, a3, v13, v12, v17);
  v53 = v55;
  v19 = v18;
  v45 = v20;
  v21 = (unsigned int)(*(_DWORD *)(a5 + 56) + 1);
  v54 = 0x800000000LL;
  v22 = v21;
  v23 = v55;
  if ( (unsigned int)v21 > 8 )
  {
    v42 = v19;
    sub_16CD150((__int64)&v53, v55, v21, 16, (int)v19, v21);
    v23 = v53;
    v19 = v42;
    v22 = v21;
  }
  v24 = 16 * v21;
  LODWORD(v54) = v22;
  for ( i = &v23[v24]; i != v23; v23 += 16 )
  {
    if ( v23 )
    {
      *(_QWORD *)v23 = 0;
      *((_DWORD *)v23 + 2) = 0;
    }
  }
  v26 = *(unsigned int *)(a5 + 60);
  v27 = *(const void **)(a5 + 40);
  v56 = dest;
  v57 = 0x800000000LL;
  v28 = 16 * v26;
  if ( v26 > 8 )
  {
    v38 = v19;
    v39 = 16 * v26;
    v41 = v27;
    v44 = v26;
    sub_16CD150((__int64)&v56, dest, v26, 16, (int)v19, v26);
    v26 = v44;
    v27 = v41;
    v28 = v39;
    v19 = v38;
    v36 = &v56[16 * (unsigned int)v57];
  }
  else
  {
    if ( !v28 )
      goto LABEL_11;
    v36 = dest;
  }
  v40 = v19;
  v43 = v26;
  memcpy(v36, v27, v28);
  v28 = (unsigned int)v57;
  v19 = v40;
  v26 = v43;
LABEL_11:
  v29 = *(_QWORD *)(a5 + 32);
  v30 = v28 + v26;
  v31 = (unsigned __int64)v53;
  LODWORD(v57) = v30;
  *(_QWORD *)v53 = *(_QWORD *)v29;
  *(_DWORD *)(v31 + 8) = *(_DWORD *)(v29 + 8);
  v32 = *(_QWORD *)(a5 + 32);
  v33 = (unsigned __int64)v53;
  *((_QWORD *)v53 + 2) = *(_QWORD *)(v32 + 40);
  *(_DWORD *)(v33 + 24) = *(_DWORD *)(v32 + 48);
  *(_QWORD *)(v33 + 32) = v47;
  *(_DWORD *)(v33 + 40) = v46;
  *(_QWORD *)(v33 + 48) = v19;
  *(_DWORD *)(v33 + 56) = v45;
  if ( *(_DWORD *)(a5 + 56) == 4 )
  {
    v34 = *(_QWORD *)(a5 + 32);
    *(_QWORD *)(v33 + 64) = *(_QWORD *)(v34 + 120);
    *(_DWORD *)(v33 + 72) = *(_DWORD *)(v34 + 128);
  }
  *((_QWORD *)&v37 + 1) = (unsigned int)v54;
  *(_QWORD *)&v37 = v33;
  result = sub_1D373B0(a7, 0x2Eu, (__int64)&v51, v56, (unsigned int)v57, *(double *)a1.m128i_i64, a2, a3, v30, v37);
  if ( v56 != dest )
  {
    v48 = result;
    _libc_free((unsigned __int64)v56);
    result = v48;
  }
  if ( v53 != v55 )
  {
    v49 = result;
    _libc_free((unsigned __int64)v53);
    result = v49;
  }
  if ( v51 )
  {
    v50 = result;
    sub_161E7C0((__int64)&v51, v51);
    return v50;
  }
  return result;
}
