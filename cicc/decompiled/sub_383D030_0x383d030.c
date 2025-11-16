// Function: sub_383D030
// Address: 0x383d030
//
__int64 *__fastcall sub_383D030(_QWORD *a1, unsigned __int64 a2, __m128i a3)
{
  int v4; // eax
  __int64 v5; // r12
  unsigned int v6; // ebx
  char v7; // r15
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _WORD *v12; // r11
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 (__fastcall **v15)(__int64, __int64, unsigned int); // rsi
  __int64 *result; // rax
  unsigned int v17; // eax
  __int16 *v18; // rdx
  __int16 v19; // ax
  __int64 v20; // rdx
  __int64 *v21; // rdx
  unsigned __int64 v22; // rax
  const __m128i *v23; // r12
  int v24; // ecx
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rdx
  const __m128i *v27; // r15
  __m128i *v28; // rax
  __int64 v29; // rbx
  unsigned __int8 *v30; // rax
  unsigned __int8 **v31; // rbx
  int v32; // edx
  _WORD *v33; // [rsp+8h] [rbp-E8h]
  int v34; // [rsp+10h] [rbp-E0h]
  int v35; // [rsp+10h] [rbp-E0h]
  __int64 *v36; // [rsp+18h] [rbp-D8h]
  __int64 v37; // [rsp+30h] [rbp-C0h] BYREF
  int v38; // [rsp+38h] [rbp-B8h]
  _OWORD v39[2]; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v40[4]; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v41; // [rsp+80h] [rbp-70h] BYREF
  __int64 v42; // [rsp+88h] [rbp-68h]
  __int64 v43; // [rsp+90h] [rbp-60h] BYREF
  __int64 v44; // [rsp+98h] [rbp-58h]
  __int64 v45; // [rsp+A0h] [rbp-50h]

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 239 )
  {
    if ( (unsigned int)(v4 - 242) > 1 )
    {
      v5 = 0;
      v7 = 0;
      v6 = 0;
      v8 = 0;
      if ( v4 != 258 )
        goto LABEL_6;
LABEL_18:
      v17 = sub_2FE5E70(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
      v9 = v17;
      if ( v17 != 729 )
      {
        v12 = (_WORD *)*a1;
        if ( *(_QWORD *)(*a1 + 8LL * (int)v17 + 525288) )
          goto LABEL_8;
      }
      v18 = *(__int16 **)(a2 + 48);
      v19 = *v18;
      v20 = *((_QWORD *)v18 + 1);
      LOWORD(v41) = v19;
      v42 = v20;
      if ( v19 )
      {
        if ( (unsigned __int16)(v19 - 17) > 0xD3u )
          goto LABEL_25;
      }
      else if ( !sub_30070B0((__int64)&v41) )
      {
        goto LABEL_25;
      }
      return (__int64 *)sub_3412A00((_QWORD *)a1[1], a2, 0, v9, v10, v11, a3);
    }
  }
  else if ( v4 <= 237 && (unsigned int)(v4 - 101) > 0x2F )
  {
    v5 = 0;
    v6 = 0;
    v7 = 0;
    v8 = 0;
    goto LABEL_5;
  }
  v21 = *(__int64 **)(a2 + 40);
  v6 = 1;
  v7 = 1;
  v5 = *v21;
  v8 = v21[1];
LABEL_5:
  if ( v4 == 109 )
    goto LABEL_18;
LABEL_6:
  LODWORD(v9) = sub_2FE5EA0(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
  if ( (_DWORD)v9 != 729 )
  {
    v12 = (_WORD *)*a1;
    if ( *(_QWORD *)(*a1 + 8LL * (int)v9 + 525288) )
    {
LABEL_8:
      v13 = *(_QWORD *)(a2 + 40);
      v41 = 0;
      LOBYTE(v45) = 5;
      v14 = *(_QWORD *)(a2 + 80);
      v42 = 0;
      v43 = 0;
      v44 = 0;
      v39[0] = _mm_loadu_si128((const __m128i *)(v13 + 40LL * v6));
      v37 = v14;
      v39[1] = _mm_loadu_si128((const __m128i *)(v13 + 40LL * (v6 + 1)));
      if ( v14 )
      {
        v33 = v12;
        v34 = v9;
        sub_B96E90((__int64)&v37, v14, 1);
        v12 = v33;
        LODWORD(v9) = v34;
      }
      v15 = *(__int64 (__fastcall ***)(__int64, __int64, unsigned int))(a2 + 48);
      v38 = *(_DWORD *)(a2 + 72);
      sub_3494590(
        (__int64)v40,
        v12,
        a1[1],
        v9,
        *(unsigned __int16 *)v15,
        v15[1],
        (__int64)v39,
        2u,
        (__int64)v41,
        v42,
        v43,
        v44,
        v45,
        (__int64)&v37,
        v5,
        v8);
      if ( v37 )
        sub_B91220((__int64)&v37, v37);
      sub_3760E70((__int64)a1, a2, 0, v40[0], v40[1]);
      if ( v7 )
        sub_3760E70((__int64)a1, a2, 1, v40[2], v40[3]);
      return 0;
    }
  }
LABEL_25:
  v22 = *(unsigned int *)(a2 + 64);
  v23 = *(const __m128i **)(a2 + 40);
  v24 = 0;
  v41 = &v43;
  v25 = 40 * v22;
  v26 = v22;
  v27 = (const __m128i *)((char *)v23 + 40 * v22);
  v42 = 0x300000000LL;
  v28 = (__m128i *)&v43;
  if ( v25 > 0x78 )
  {
    v35 = v26;
    sub_C8D5F0((__int64)&v41, &v43, v26, 0x10u, v10, v11);
    v24 = v42;
    LODWORD(v26) = v35;
    v28 = (__m128i *)&v41[2 * (unsigned int)v42];
  }
  if ( v23 != v27 )
  {
    do
    {
      if ( v28 )
        *v28 = _mm_loadu_si128(v23);
      v23 = (const __m128i *)((char *)v23 + 40);
      ++v28;
    }
    while ( v27 != v23 );
    v24 = v42;
  }
  v29 = v6 + 1;
  LODWORD(v42) = v24 + v26;
  v30 = sub_383B380(
          (__int64)a1,
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40 * v29),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40 * v29 + 8));
  v31 = (unsigned __int8 **)&v41[2 * v29];
  *v31 = v30;
  *((_DWORD *)v31 + 2) = v32;
  result = sub_33EC210((_QWORD *)a1[1], (__int64 *)a2, (__int64)v41, (unsigned int)v42);
  if ( v41 != &v43 )
  {
    v36 = result;
    _libc_free((unsigned __int64)v41);
    return v36;
  }
  return result;
}
