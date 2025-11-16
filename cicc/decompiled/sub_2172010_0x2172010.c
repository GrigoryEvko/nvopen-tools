// Function: sub_2172010
// Address: 0x2172010
//
__int64 __fastcall sub_2172010(__int64 a1, double a2, double a3, double a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // r12
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // r13
  unsigned int v12; // edx
  __int64 v13; // r12
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned int v19; // edx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rdx
  char v24; // di
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  const void **v29; // r9
  __int64 v30; // r8
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rdx
  __m128i v35; // xmm2
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  char v40; // di
  __int64 v41; // rax
  unsigned int v42; // eax
  __int64 v43; // [rsp+8h] [rbp-E8h]
  const void **v44; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v45; // [rsp+48h] [rbp-A8h] BYREF
  _OWORD v46[3]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+80h] [rbp-70h]
  __int64 v48; // [rsp+88h] [rbp-68h]
  __m128i v49; // [rsp+90h] [rbp-60h]
  __m128i v50; // [rsp+A0h] [rbp-50h]
  __m128i v51; // [rsp+B0h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 32);
  v9 = *(_QWORD *)(*(_QWORD *)(v8 + 200) + 88LL);
  v10 = *(_DWORD *)(v9 + 32);
  if ( v10 > 0x40 )
  {
    if ( v10 == (unsigned int)sub_16A57B0(v9 + 24) )
      goto LABEL_3;
LABEL_8:
    v11 = sub_2170FD0(*(_QWORD *)(v8 + 120), *(_DWORD *)(v8 + 128), &v45);
    v13 = v19;
    if ( v11 )
      goto LABEL_4;
    v20 = *(_QWORD *)(a1 + 32);
    v21 = *(_QWORD *)(v20 + 120);
    if ( *(_WORD *)(v21 + 24) != 50 )
      return a1;
    v22 = *(_QWORD *)(v21 + 32);
    if ( *(_WORD *)(*(_QWORD *)v22 + 24LL) != 48 )
      return a1;
    v23 = *(_QWORD *)(v21 + 40) + 16LL * *(unsigned int *)(v20 + 128);
    v24 = *(_BYTE *)v23;
    v25 = *(_QWORD *)(v23 + 8);
    LOBYTE(v46[0]) = v24;
    *((_QWORD *)&v46[0] + 1) = v25;
    if ( v24 )
      v26 = sub_216FFF0(v24);
    else
      v26 = sub_1F58D40((__int64)v46);
    v11 = *(_QWORD *)(v22 + 40);
    v13 = *(unsigned int *)(v22 + 48);
    v45 = v26 >> 1;
LABEL_25:
    if ( !v11 )
      return a1;
    goto LABEL_4;
  }
  if ( *(_QWORD *)(v9 + 24) )
    goto LABEL_8;
LABEL_3:
  v11 = sub_2171180(*(_QWORD *)(v8 + 120), *(_QWORD *)(v8 + 128), &v45);
  v13 = v12;
  if ( !v11 )
  {
    v36 = *(_QWORD *)(a1 + 32);
    v37 = *(_QWORD *)(v36 + 120);
    if ( *(_WORD *)(v37 + 24) != 50 )
      return a1;
    v38 = *(_QWORD *)(v37 + 32);
    if ( *(_WORD *)(*(_QWORD *)(v38 + 40) + 24LL) != 48 )
      return a1;
    v39 = *(_QWORD *)(v37 + 40) + 16LL * *(unsigned int *)(v36 + 128);
    v40 = *(_BYTE *)v39;
    v41 = *(_QWORD *)(v39 + 8);
    LOBYTE(v46[0]) = v40;
    *((_QWORD *)&v46[0] + 1) = v41;
    if ( v40 )
      v42 = sub_216FFF0(v40);
    else
      v42 = sub_1F58D40((__int64)v46);
    v45 = v42 >> 1;
    v11 = *(_QWORD *)v38;
    v13 = *(unsigned int *)(v38 + 8);
    goto LABEL_25;
  }
LABEL_4:
  v14 = *(_QWORD *)(a1 + 40);
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  LOBYTE(v46[0]) = v15;
  *((_QWORD *)&v46[0] + 1) = v16;
  if ( v15 )
  {
    if ( (unsigned int)sub_216FFF0(v15) >> 1 != v45 )
      return a1;
  }
  else
  {
    v17 = sub_1F58D40((__int64)v46);
    if ( v17 >> 1 != v45 )
      return a1;
  }
  v27 = *(_QWORD *)(a1 + 72);
  v28 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 120LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 32) + 128LL));
  v29 = (const void **)*((_QWORD *)v28 + 1);
  v30 = *v28;
  *(_QWORD *)&v46[0] = v27;
  if ( v27 )
  {
    v43 = v30;
    v44 = v29;
    sub_1623A60((__int64)v46, v27, 2);
    v30 = v43;
    v29 = v44;
  }
  DWORD2(v46[0]) = *(_DWORD *)(a1 + 64);
  v31 = sub_1D321C0(a6, v11, v13, (__int64)v46, v30, v29, a2, a3, a4);
  v33 = v32;
  if ( *(_QWORD *)&v46[0] )
    sub_161E7C0((__int64)v46, *(__int64 *)&v46[0]);
  v34 = *(_QWORD *)(a1 + 32);
  v46[0] = _mm_loadu_si128((const __m128i *)v34);
  v46[1] = _mm_loadu_si128((const __m128i *)(v34 + 40));
  v35 = _mm_loadu_si128((const __m128i *)(v34 + 80));
  v47 = v31;
  v48 = v33;
  v46[2] = v35;
  v49 = _mm_loadu_si128((const __m128i *)(v34 + 160));
  v50 = _mm_loadu_si128((const __m128i *)(v34 + 200));
  v51 = _mm_loadu_si128((const __m128i *)(v34 + 240));
  sub_1D2E160(a6, (__int64 *)a1, (__int64)v46, 7);
  return a1;
}
