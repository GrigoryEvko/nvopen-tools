// Function: sub_234D390
// Address: 0x234d390
//
__int64 __fastcall sub_234D390(__int64 a1, __int64 *a2, char a3, char a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 v17; // rdi
  void *v18; // rsi
  __m128i v19; // xmm3
  unsigned __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // rbx
  _QWORD *v27; // r15
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v33; // [rsp+20h] [rbp-1E0h]
  __int64 v34; // [rsp+28h] [rbp-1D8h]
  __int64 v35; // [rsp+30h] [rbp-1D0h]
  __int64 v36; // [rsp+38h] [rbp-1C8h]
  __int64 v37; // [rsp+40h] [rbp-1C0h]
  __int64 v38; // [rsp+48h] [rbp-1B8h]
  unsigned __int64 v39; // [rsp+50h] [rbp-1B0h]
  __m128i v40; // [rsp+58h] [rbp-1A8h] BYREF
  __m128i v41; // [rsp+68h] [rbp-198h] BYREF
  __int16 v42; // [rsp+78h] [rbp-188h]
  _BYTE v43[8]; // [rsp+80h] [rbp-180h] BYREF
  unsigned __int64 v44; // [rsp+88h] [rbp-178h]
  char v45; // [rsp+9Ch] [rbp-164h]
  _BYTE v46[128]; // [rsp+A0h] [rbp-160h] BYREF
  _BYTE v47[8]; // [rsp+120h] [rbp-E0h] BYREF
  unsigned __int64 v48; // [rsp+128h] [rbp-D8h]
  char v49; // [rsp+13Ch] [rbp-C4h]
  _BYTE v50[128]; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+1C0h] [rbp-40h]

  v33 = *a2;
  v34 = a2[1];
  v7 = a2[2];
  v40 = _mm_loadu_si128((const __m128i *)(a2 + 7));
  v35 = v7;
  v8 = a2[3];
  v41 = _mm_loadu_si128((const __m128i *)(a2 + 9));
  v36 = v8;
  v37 = a2[4];
  v38 = a2[5];
  v9 = a2[6];
  a2[6] = 0;
  v39 = v9;
  v42 = *((_WORD *)a2 + 44);
  sub_C8CF70((__int64)v43, v46, 16, (__int64)(a2 + 16), (__int64)(a2 + 12));
  v10 = v50;
  sub_C8CF70((__int64)v47, v50, 16, (__int64)(a2 + 36), (__int64)(a2 + 32));
  v51 = a2[52];
  v11 = sub_22077B0(0x1B0u);
  v16 = v11;
  if ( v11 )
  {
    v17 = v11 + 104;
    v18 = (void *)(v11 + 136);
    v19 = _mm_loadu_si128(&v41);
    *(__m128i *)(v11 + 64) = _mm_loadu_si128(&v40);
    *(_QWORD *)v11 = &unk_4A0FAF8;
    *(__m128i *)(v11 + 80) = v19;
    *(_QWORD *)(v11 + 8) = v33;
    *(_QWORD *)(v11 + 16) = v34;
    *(_QWORD *)(v11 + 24) = v35;
    *(_QWORD *)(v11 + 32) = v36;
    *(_QWORD *)(v11 + 40) = v37;
    *(_QWORD *)(v11 + 48) = v38;
    v20 = v39;
    v39 = 0;
    *(_QWORD *)(v16 + 56) = v20;
    *(_WORD *)(v16 + 96) = v42;
    sub_C8CF70(v17, v18, 16, (__int64)v46, (__int64)v43);
    v10 = (_BYTE *)(v16 + 296);
    sub_C8CF70(v16 + 264, (void *)(v16 + 296), 16, (__int64)v50, (__int64)v47);
    *(_QWORD *)(v16 + 424) = v51;
  }
  v21 = v49 == 0;
  *(_QWORD *)a1 = v16;
  *(_BYTE *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 9) = a4;
  if ( v21 )
    _libc_free(v48);
  if ( !v45 )
    _libc_free(v44);
  if ( v39 )
  {
    sub_FFCE90(v39, (__int64)v10, v12, v13, v14, v15);
    sub_FFD870(v39, (__int64)v10, v22, v23, v24, v25);
    sub_FFBC40(v39, (__int64)v10);
    v26 = *(_QWORD **)(v39 + 680);
    v27 = *(_QWORD **)(v39 + 672);
    if ( v26 != v27 )
    {
      do
      {
        v28 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v27[7];
        *v27 = &unk_49E5048;
        if ( v28 )
          v28(v27 + 5, v27 + 5, 3);
        *v27 = &unk_49DB368;
        v29 = v27[3];
        if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
          sub_BD60C0(v27 + 1);
        v27 += 9;
      }
      while ( v26 != v27 );
      v27 = *(_QWORD **)(v39 + 672);
    }
    if ( v27 )
      j_j___libc_free_0((unsigned __int64)v27);
    if ( *(_BYTE *)(v39 + 596) )
    {
      v30 = *(_QWORD *)v39;
      if ( *(_QWORD *)v39 == v39 + 16 )
      {
LABEL_21:
        j_j___libc_free_0(v39);
        return a1;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v39 + 576));
      v30 = *(_QWORD *)v39;
      if ( *(_QWORD *)v39 == v39 + 16 )
        goto LABEL_21;
    }
    _libc_free(v30);
    goto LABEL_21;
  }
  return a1;
}
