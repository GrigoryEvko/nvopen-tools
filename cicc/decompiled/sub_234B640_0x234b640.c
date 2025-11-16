// Function: sub_234B640
// Address: 0x234b640
//
__int64 __fastcall sub_234B640(__int64 a1, __int64 *a2, char a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // rdi
  void *v17; // rsi
  __m128i v18; // xmm3
  unsigned __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rbx
  _QWORD *v26; // r15
  void (__fastcall *v27)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __int64 v31; // [rsp+10h] [rbp-1E0h]
  __int64 v32; // [rsp+18h] [rbp-1D8h]
  __int64 v33; // [rsp+20h] [rbp-1D0h]
  __int64 v34; // [rsp+28h] [rbp-1C8h]
  __int64 v35; // [rsp+30h] [rbp-1C0h]
  __int64 v36; // [rsp+38h] [rbp-1B8h]
  unsigned __int64 v37; // [rsp+40h] [rbp-1B0h]
  __m128i v38; // [rsp+48h] [rbp-1A8h] BYREF
  __m128i v39; // [rsp+58h] [rbp-198h] BYREF
  __int16 v40; // [rsp+68h] [rbp-188h]
  _BYTE v41[8]; // [rsp+70h] [rbp-180h] BYREF
  unsigned __int64 v42; // [rsp+78h] [rbp-178h]
  char v43; // [rsp+8Ch] [rbp-164h]
  _BYTE v44[128]; // [rsp+90h] [rbp-160h] BYREF
  _BYTE v45[8]; // [rsp+110h] [rbp-E0h] BYREF
  unsigned __int64 v46; // [rsp+118h] [rbp-D8h]
  char v47; // [rsp+12Ch] [rbp-C4h]
  _BYTE v48[128]; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+1B0h] [rbp-40h]

  v31 = *a2;
  v6 = a2[1];
  v38 = _mm_loadu_si128((const __m128i *)(a2 + 7));
  v32 = v6;
  v7 = a2[2];
  v39 = _mm_loadu_si128((const __m128i *)(a2 + 9));
  v33 = v7;
  v34 = a2[3];
  v35 = a2[4];
  v36 = a2[5];
  v8 = a2[6];
  a2[6] = 0;
  v37 = v8;
  v40 = *((_WORD *)a2 + 44);
  sub_C8CF70((__int64)v41, v44, 16, (__int64)(a2 + 16), (__int64)(a2 + 12));
  v9 = v48;
  sub_C8CF70((__int64)v45, v48, 16, (__int64)(a2 + 36), (__int64)(a2 + 32));
  v49 = a2[52];
  v10 = sub_22077B0(0x1B0u);
  v15 = v10;
  if ( v10 )
  {
    v16 = v10 + 104;
    v17 = (void *)(v10 + 136);
    v18 = _mm_loadu_si128(&v39);
    *(__m128i *)(v10 + 64) = _mm_loadu_si128(&v38);
    *(_QWORD *)v10 = &unk_4A0FAF8;
    *(__m128i *)(v10 + 80) = v18;
    *(_QWORD *)(v10 + 8) = v31;
    *(_QWORD *)(v10 + 16) = v32;
    *(_QWORD *)(v10 + 24) = v33;
    *(_QWORD *)(v10 + 32) = v34;
    *(_QWORD *)(v10 + 40) = v35;
    *(_QWORD *)(v10 + 48) = v36;
    v19 = v37;
    v37 = 0;
    *(_QWORD *)(v15 + 56) = v19;
    *(_WORD *)(v15 + 96) = v40;
    sub_C8CF70(v16, v17, 16, (__int64)v44, (__int64)v41);
    v9 = (_BYTE *)(v15 + 296);
    sub_C8CF70(v15 + 264, (void *)(v15 + 296), 16, (__int64)v48, (__int64)v45);
    *(_QWORD *)(v15 + 424) = v49;
  }
  v20 = v47 == 0;
  *(_QWORD *)a1 = v15;
  *(_BYTE *)(a1 + 8) = a3;
  if ( v20 )
    _libc_free(v46);
  if ( !v43 )
    _libc_free(v42);
  if ( v37 )
  {
    sub_FFCE90(v37, (__int64)v9, v11, v12, v13, v14);
    sub_FFD870(v37, (__int64)v9, v21, v22, v23, v24);
    sub_FFBC40(v37, (__int64)v9);
    v25 = *(_QWORD **)(v37 + 680);
    v26 = *(_QWORD **)(v37 + 672);
    if ( v25 != v26 )
    {
      do
      {
        v27 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v26[7];
        *v26 = &unk_49E5048;
        if ( v27 )
          v27(v26 + 5, v26 + 5, 3);
        *v26 = &unk_49DB368;
        v28 = v26[3];
        if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
          sub_BD60C0(v26 + 1);
        v26 += 9;
      }
      while ( v25 != v26 );
      v26 = *(_QWORD **)(v37 + 672);
    }
    if ( v26 )
      j_j___libc_free_0((unsigned __int64)v26);
    if ( *(_BYTE *)(v37 + 596) )
    {
      v29 = *(_QWORD *)v37;
      if ( *(_QWORD *)v37 == v37 + 16 )
      {
LABEL_21:
        j_j___libc_free_0(v37);
        return a1;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v37 + 576));
      v29 = *(_QWORD *)v37;
      if ( *(_QWORD *)v37 == v37 + 16 )
        goto LABEL_21;
    }
    _libc_free(v29);
    goto LABEL_21;
  }
  return a1;
}
