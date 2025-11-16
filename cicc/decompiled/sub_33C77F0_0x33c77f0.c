// Function: sub_33C77F0
// Address: 0x33c77f0
//
void __fastcall sub_33C77F0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  int v11; // eax
  char v12; // dl
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __m128i v19; // xmm1
  __int64 (__fastcall *v20)(_QWORD *, _QWORD *, int); // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // edx
  unsigned __int8 **v24; // rcx
  unsigned __int8 *v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  _QWORD *v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // r14
  __m128i si128; // xmm0
  __int64 v35; // rax
  _WORD *v36; // rdi
  unsigned __int8 *v37; // rsi
  size_t v38; // r12
  unsigned __int64 v39; // rax
  int v40; // ecx
  int v41; // r9d
  __int64 v42; // rax
  __int64 v43; // rax
  char v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  unsigned __int8 **v46; // [rsp+8h] [rbp-68h]
  char v47; // [rsp+1Fh] [rbp-51h] BYREF
  __m128i v48; // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v49)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-40h]
  _BYTE *(__fastcall *v50)(_QWORD *); // [rsp+38h] [rbp-38h]

  sub_3387170(a1, (__int64)a2);
  v7 = *a2;
  if ( (unsigned int)(v7 - 30) <= 0xA )
  {
    sub_33C6820(a1, *((_QWORD *)a2 + 5));
    if ( *a2 != 85 )
      goto LABEL_3;
  }
  else if ( (_BYTE)v7 != 85 )
  {
    goto LABEL_3;
  }
  v27 = *((_QWORD *)a2 - 4);
  if ( !v27
    || *(_BYTE *)v27
    || (v4 = *((_QWORD *)a2 + 10), *(_QWORD *)(v27 + 24) != v4)
    || (*(_BYTE *)(v27 + 33) & 0x20) == 0
    || (unsigned int)(*(_DWORD *)(v27 + 36) - 68) > 3 )
  {
LABEL_3:
    ++*(_DWORD *)(a1 + 848);
    *(_QWORD *)a1 = a2;
    v47 = 0;
    if ( (a2[7] & 0x20) != 0 )
      goto LABEL_4;
LABEL_34:
    v9 = 0;
    goto LABEL_6;
  }
  *(_QWORD *)a1 = a2;
  v47 = 0;
  if ( (a2[7] & 0x20) == 0 )
    goto LABEL_34;
LABEL_4:
  v8 = sub_B91C10((__int64)a2, 37);
  v9 = v8;
  if ( (a2[7] & 0x20) == 0 )
  {
    v10 = 0;
    if ( v8 )
      goto LABEL_17;
LABEL_6:
    sub_3389AA0(a1, *a2 - 29, (__int64)a2, v4, v5, v6);
    v11 = *a2;
    if ( (unsigned int)(v11 - 30) <= 0xA || (v12 = *(_BYTE *)(a1 + 1016)) != 0 )
    {
      *(_QWORD *)a1 = 0;
      return;
    }
    v10 = 0;
    v13 = 0;
LABEL_11:
    if ( (unsigned __int8)v11 <= 0x1Cu
      || (v28 = (unsigned int)(v11 - 34), (unsigned __int8)v28 > 0x33u)
      || (v29 = 0x8000000000041LL, !_bittest64(&v29, v28))
      || (v30 = *((_QWORD *)a2 - 4)) == 0
      || *(_BYTE *)v30
      || *(_QWORD *)(v30 + 24) != *((_QWORD *)a2 + 10)
      || *(_DWORD *)(v30 + 36) != 151 )
    {
      v44 = v12;
      sub_33BFCB0(a1, (__int64)a2);
      v12 = v44;
    }
    if ( !v12 )
      goto LABEL_14;
    goto LABEL_20;
  }
  v10 = sub_B91C10((__int64)a2, 40);
  if ( !(v9 | v10) )
    goto LABEL_6;
LABEL_17:
  v14 = *(_QWORD *)(a1 + 864);
  v48.m128i_i64[0] = (__int64)&v47;
  v50 = sub_3364FF0;
  v45 = v14;
  v49 = sub_3365300;
  v13 = sub_22077B0(0x38u);
  if ( v13 )
  {
    v18 = *(_QWORD *)(v45 + 768);
    *(_QWORD *)(v13 + 16) = v45;
    *(_QWORD *)(v13 + 8) = v18;
    *(_QWORD *)(v45 + 768) = v13;
    v19 = _mm_loadu_si128(&v48);
    *(_QWORD *)v13 = &unk_4A36708;
    v20 = v49;
    *(__m128i *)(v13 + 24) = v19;
    *(_QWORD *)(v13 + 40) = v20;
    *(_QWORD *)(v13 + 48) = v50;
  }
  else if ( v49 )
  {
    v49(&v48, &v48, 3);
  }
  sub_3389AA0(a1, *a2 - 29, (__int64)a2, v15, v16, v17);
  v11 = *a2;
  if ( (unsigned int)(v11 - 30) > 0xA )
  {
    v12 = 1;
    if ( !*(_BYTE *)(a1 + 1016) )
      goto LABEL_11;
  }
LABEL_20:
  v21 = *(unsigned int *)(a1 + 32);
  v22 = *(_QWORD *)(a1 + 16);
  if ( !(_DWORD)v21 )
    goto LABEL_46;
  v23 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = (unsigned __int8 **)(v22 + 24LL * v23);
  v25 = *v24;
  if ( a2 != *v24 )
  {
    v40 = 1;
    while ( v25 != (unsigned __int8 *)-4096LL )
    {
      v41 = v40 + 1;
      v23 = (v21 - 1) & (v40 + v23);
      v24 = (unsigned __int8 **)(v22 + 24LL * v23);
      v25 = *v24;
      if ( a2 == *v24 )
        goto LABEL_22;
      v40 = v41;
    }
    goto LABEL_46;
  }
LABEL_22:
  if ( v24 == (unsigned __int8 **)(v22 + 24 * v21) )
  {
LABEL_46:
    if ( v47 )
    {
      v31 = sub_CB72A0();
      v32 = (__m128i *)v31[4];
      v33 = (__int64)v31;
      if ( v31[3] - (_QWORD)v32 <= 0x33u )
      {
        v33 = sub_CB6200((__int64)v31, "warning: loosing !pcsections and/or !mmra metadata [", 0x34u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44DC2B0);
        v32[3].m128i_i32[0] = 1528848756;
        *v32 = si128;
        v32[1] = _mm_load_si128((const __m128i *)&xmmword_44DC2C0);
        v32[2] = _mm_load_si128((const __m128i *)&xmmword_44DC2D0);
        v31[4] += 52LL;
      }
      v35 = sub_B43CA0((__int64)a2);
      v36 = *(_WORD **)(v33 + 32);
      v37 = *(unsigned __int8 **)(v35 + 168);
      v38 = *(_QWORD *)(v35 + 176);
      v39 = *(_QWORD *)(v33 + 24) - (_QWORD)v36;
      if ( v38 > v39 )
      {
        v42 = sub_CB6200(v33, v37, v38);
        v36 = *(_WORD **)(v42 + 32);
        v33 = v42;
        v39 = *(_QWORD *)(v42 + 24) - (_QWORD)v36;
      }
      else if ( v38 )
      {
        memcpy(v36, v37, v38);
        v43 = *(_QWORD *)(v33 + 24);
        v36 = (_WORD *)(v38 + *(_QWORD *)(v33 + 32));
        *(_QWORD *)(v33 + 32) = v36;
        v39 = v43 - (_QWORD)v36;
      }
      if ( v39 <= 1 )
      {
        sub_CB6200(v33, (unsigned __int8 *)"]\n", 2u);
      }
      else
      {
        *v36 = 2653;
        *(_QWORD *)(v33 + 32) += 2LL;
      }
    }
    goto LABEL_14;
  }
  if ( v9 )
  {
    v46 = v24;
    v48.m128i_i64[0] = (__int64)v24[1];
    v26 = sub_337D790(*(_QWORD *)(a1 + 864) + 728LL, v48.m128i_i64);
    v24 = v46;
    v26[4] = v9;
  }
  if ( v10 )
  {
    v48.m128i_i64[0] = (__int64)v24[1];
    sub_337D790(*(_QWORD *)(a1 + 864) + 728LL, v48.m128i_i64)[5] = v10;
  }
LABEL_14:
  *(_QWORD *)a1 = 0;
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
}
