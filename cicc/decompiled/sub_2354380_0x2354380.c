// Function: sub_2354380
// Address: 0x2354380
//
void __fastcall sub_2354380(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __m128i *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdi
  __m128i v9; // xmm3
  unsigned __int64 *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  void (__fastcall *v24)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  __int64 v27; // [rsp+18h] [rbp-1E8h] BYREF
  __int64 v28; // [rsp+20h] [rbp-1E0h]
  __int64 v29; // [rsp+28h] [rbp-1D8h]
  __int64 v30; // [rsp+30h] [rbp-1D0h]
  __int64 v31; // [rsp+38h] [rbp-1C8h]
  __int64 v32; // [rsp+40h] [rbp-1C0h]
  __int64 v33; // [rsp+48h] [rbp-1B8h]
  __int64 v34; // [rsp+50h] [rbp-1B0h]
  __m128i v35; // [rsp+58h] [rbp-1A8h] BYREF
  __m128i v36; // [rsp+68h] [rbp-198h] BYREF
  __int16 v37; // [rsp+78h] [rbp-188h]
  _BYTE v38[8]; // [rsp+80h] [rbp-180h] BYREF
  unsigned __int64 v39; // [rsp+88h] [rbp-178h]
  char v40; // [rsp+9Ch] [rbp-164h]
  _BYTE v41[128]; // [rsp+A0h] [rbp-160h] BYREF
  _BYTE v42[8]; // [rsp+120h] [rbp-E0h] BYREF
  unsigned __int64 v43; // [rsp+128h] [rbp-D8h]
  char v44; // [rsp+13Ch] [rbp-C4h]
  _BYTE v45[128]; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+1C0h] [rbp-40h]

  v28 = *a2;
  v3 = a2[1];
  v35 = _mm_loadu_si128((const __m128i *)(a2 + 7));
  v29 = v3;
  v4 = a2[2];
  v36 = _mm_loadu_si128((const __m128i *)(a2 + 9));
  v30 = v4;
  v31 = a2[3];
  v32 = a2[4];
  v33 = a2[5];
  v5 = a2[6];
  a2[6] = 0;
  v34 = v5;
  v37 = *((_WORD *)a2 + 44);
  sub_C8CF70((__int64)v38, v41, 16, (__int64)(a2 + 16), (__int64)(a2 + 12));
  sub_C8CF70((__int64)v42, v45, 16, (__int64)(a2 + 36), (__int64)(a2 + 32));
  v46 = a2[52];
  v6 = (__m128i *)sub_22077B0(0x1B0u);
  v7 = (__int64)v6;
  if ( v6 )
  {
    v8 = (__int64)&v6[6].m128i_i64[1];
    v9 = _mm_loadu_si128(&v36);
    v10 = &v6[8].m128i_u64[1];
    v6[4] = _mm_loadu_si128(&v35);
    v6->m128i_i64[0] = (__int64)&unk_4A0FAF8;
    v11 = v28;
    *(__m128i *)(v7 + 80) = v9;
    *(_QWORD *)(v7 + 8) = v11;
    *(_QWORD *)(v7 + 16) = v29;
    *(_QWORD *)(v7 + 24) = v30;
    *(_QWORD *)(v7 + 32) = v31;
    *(_QWORD *)(v7 + 40) = v32;
    *(_QWORD *)(v7 + 48) = v33;
    v12 = v34;
    v34 = 0;
    *(_QWORD *)(v7 + 56) = v12;
    *(_WORD *)(v7 + 96) = v37;
    sub_C8CF70(v8, v10, 16, (__int64)v41, (__int64)v38);
    sub_C8CF70(v7 + 264, (void *)(v7 + 296), 16, (__int64)v45, (__int64)v42);
    *(_QWORD *)(v7 + 424) = v46;
  }
  v27 = v7;
  sub_2353900(a1, (unsigned __int64 *)&v27);
  sub_233EFE0(&v27);
  if ( !v44 )
    _libc_free(v43);
  if ( !v40 )
    _libc_free(v39);
  v17 = v34;
  if ( v34 )
  {
    sub_FFCE90(v34, (__int64)&v27, v13, v14, v15, v16);
    sub_FFD870(v17, (__int64)&v27, v18, v19, v20, v21);
    sub_FFBC40(v17, (__int64)&v27);
    v22 = *(_QWORD **)(v17 + 680);
    v23 = *(_QWORD **)(v17 + 672);
    if ( v22 != v23 )
    {
      do
      {
        v24 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v23[7];
        *v23 = &unk_49E5048;
        if ( v24 )
          v24(v23 + 5, v23 + 5, 3);
        *v23 = &unk_49DB368;
        v25 = v23[3];
        if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
          sub_BD60C0(v23 + 1);
        v23 += 9;
      }
      while ( v22 != v23 );
      v23 = *(_QWORD **)(v17 + 672);
    }
    if ( v23 )
      j_j___libc_free_0((unsigned __int64)v23);
    if ( *(_BYTE *)(v17 + 596) )
    {
      v26 = *(_QWORD *)v17;
      if ( *(_QWORD *)v17 == v17 + 16 )
      {
LABEL_21:
        j_j___libc_free_0(v17);
        return;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v17 + 576));
      v26 = *(_QWORD *)v17;
      if ( *(_QWORD *)v17 == v17 + 16 )
        goto LABEL_21;
    }
    _libc_free(v26);
    goto LABEL_21;
  }
}
