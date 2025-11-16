// Function: sub_2ABEEB0
// Address: 0x2abeeb0
//
__int64 __fastcall sub_2ABEEB0(__int64 a1, _BYTE *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  _BYTE *v10; // r13
  _BYTE *v11; // r9
  _BYTE *v12; // rax
  _BYTE *v13; // r9
  __int64 v14; // rax
  unsigned int v15; // r12d
  _BYTE *v17; // rdx
  _BYTE *v18; // rdx
  _BYTE *v19; // rdi
  __int64 (__fastcall *v20)(unsigned __int64 *, const __m128i **, int); // r13
  __int64 v21; // rdx
  __int64 *v22; // rax
  _QWORD *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  const __m128i *v31; // r8
  __m128i *v32; // rax
  _BYTE *v33; // rdx
  __int64 v34; // r13
  const void *v35; // rsi
  __int64 *v36; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v37; // [rsp+10h] [rbp-C0h]
  bool (__fastcall *v38)(__int64, __int64 *); // [rsp+18h] [rbp-B8h]
  _BYTE *v39; // [rsp+20h] [rbp-B0h]
  _BYTE *v40; // [rsp+28h] [rbp-A8h] BYREF
  int v41; // [rsp+38h] [rbp-98h] BYREF
  int v42; // [rsp+3Ch] [rbp-94h] BYREF
  __int64 *v43; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v44; // [rsp+48h] [rbp-88h] BYREF
  __int64 v45; // [rsp+50h] [rbp-80h] BYREF
  __int64 v46; // [rsp+58h] [rbp-78h] BYREF
  __int64 v47; // [rsp+60h] [rbp-70h]
  __int64 v48; // [rsp+68h] [rbp-68h]
  __int64 *v49; // [rsp+70h] [rbp-60h] BYREF
  _QWORD *v50; // [rsp+78h] [rbp-58h]
  __int64 (__fastcall *v51)(unsigned __int64 *, const __m128i **, int); // [rsp+80h] [rbp-50h]
  bool (__fastcall *v52)(__int64, __int64 *); // [rsp+88h] [rbp-48h]
  int v53; // [rsp+90h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 40);
  v40 = a2;
  if ( !(unsigned __int8)sub_B19060(*(_QWORD *)(v9 + 416) + 56LL, a3[5], (__int64)a3, a4)
    || (unsigned int)*(unsigned __int8 *)a3 - 42 > 0x11 )
  {
    return 0;
  }
  v10 = (_BYTE *)*(a3 - 8);
  v11 = (_BYTE *)*(a3 - 4);
  v43 = a3;
  if ( v40 == v10 )
  {
    v10 = v11;
    v39 = v40;
    if ( *v11 <= 0x1Cu )
      return 0;
  }
  else
  {
    v39 = v11;
    if ( *v10 <= 0x1Cu )
      return 0;
  }
  if ( !(unsigned __int8)sub_2ABEEB0(a1, v40, v10, a4, a5) )
  {
    v12 = v40;
    v13 = v39;
    goto LABEL_7;
  }
  v12 = *(_BYTE **)(*(_QWORD *)a5 + 40LL * *(unsigned int *)(a5 + 8) - 40);
  v40 = v12;
  v10 = (_BYTE *)*(v43 - 8);
  v13 = (_BYTE *)*(v43 - 4);
  if ( v12 != v10 )
  {
LABEL_7:
    if ( v13 == v12 )
      goto LABEL_8;
    return 0;
  }
  v10 = (_BYTE *)*(v43 - 4);
LABEL_8:
  if ( (unsigned __int8)(*v10 - 42) > 0x11u )
    return 0;
  v14 = *((_QWORD *)v10 + 2);
  v44 = v10;
  if ( !v14 || *(_QWORD *)(v14 + 8) )
    return 0;
  v49 = 0;
  v50 = &v44;
  if ( *v10 == 44 )
  {
    if ( (unsigned __int8)sub_10081F0(&v49, *((_QWORD *)v10 - 8)) )
    {
      v33 = (_BYTE *)*((_QWORD *)v10 - 4);
      if ( (unsigned __int8)(*v33 - 42) <= 0x11u )
        *v50 = v33;
    }
    v10 = v44;
  }
  v17 = (_BYTE *)*((_QWORD *)v10 - 8);
  if ( *v17 != 68 && *v17 != 69 )
    return 0;
  if ( !*((_QWORD *)v17 - 4) )
    return 0;
  v45 = *((_QWORD *)v17 - 4);
  v18 = (_BYTE *)*((_QWORD *)v10 - 4);
  if ( *v18 != 68 && *v18 != 69 )
    return 0;
  if ( !*((_QWORD *)v18 - 4) )
    return 0;
  v46 = *((_QWORD *)v18 - 4);
  v19 = (_BYTE *)*((_QWORD *)v10 - 8);
  v20 = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))*((_QWORD *)v10 - 4);
  v41 = sub_DFBCA0(v19);
  v42 = sub_DFBCA0(v20);
  v38 = (bool (__fastcall *)(__int64, __int64 *))v44;
  v47 = sub_BCAE30(*((_QWORD *)v40 + 1));
  v48 = v21;
  v22 = (__int64 *)sub_BCAE30(*(_QWORD *)(v45 + 8));
  v51 = 0;
  v49 = v22;
  v37 = v47;
  v50 = v23;
  v36 = v22;
  v24 = (__int64 *)sub_22077B0(0x40u);
  if ( v24 )
  {
    *v24 = a1;
    v24[1] = (__int64)&v43;
    v24[2] = (__int64)&v45;
    v24[3] = (__int64)&v46;
    v24[4] = (__int64)&v40;
    v24[5] = (__int64)&v41;
    v24[6] = (__int64)&v42;
    v24[7] = (__int64)&v44;
  }
  v49 = v24;
  v52 = sub_2AA7AA0;
  v51 = sub_2AA87F0;
  v15 = sub_2BF1270(&v49, a4);
  sub_A17130((__int64)&v49);
  if ( !(_BYTE)v15 )
    return 0;
  v27 = *(unsigned int *)(a5 + 12);
  v49 = a3;
  v51 = v20;
  v50 = v19;
  v52 = v38;
  v53 = v37 / (unsigned __int64)v36;
  v28 = *(unsigned int *)(a5 + 8);
  v29 = v28 + 1;
  if ( v28 + 1 > v27 )
  {
    v34 = *(_QWORD *)a5;
    v35 = (const void *)(a5 + 16);
    if ( *(_QWORD *)a5 > (unsigned __int64)&v49 || (unsigned __int64)&v49 >= v34 + 40 * v28 )
    {
      sub_C8D5F0(a5, v35, v29, 0x28u, v25, v26);
      v28 = *(unsigned int *)(a5 + 8);
      v30 = *(_QWORD *)a5;
      v31 = (const __m128i *)&v49;
    }
    else
    {
      sub_C8D5F0(a5, v35, v29, 0x28u, v25, v26);
      v30 = *(_QWORD *)a5;
      v28 = *(unsigned int *)(a5 + 8);
      v31 = (const __m128i *)((char *)&v49 + *(_QWORD *)a5 - v34);
    }
  }
  else
  {
    v30 = *(_QWORD *)a5;
    v31 = (const __m128i *)&v49;
  }
  v32 = (__m128i *)(v30 + 40 * v28);
  *v32 = _mm_loadu_si128(v31);
  v32[1] = _mm_loadu_si128(v31 + 1);
  v32[2].m128i_i64[0] = v31[2].m128i_i64[0];
  ++*(_DWORD *)(a5 + 8);
  return v15;
}
