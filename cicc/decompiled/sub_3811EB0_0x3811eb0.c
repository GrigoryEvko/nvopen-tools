// Function: sub_3811EB0
// Address: 0x3811eb0
//
unsigned __int8 *__fastcall sub_3811EB0(__int64 a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  unsigned __int64 *v5; // rdx
  unsigned __int16 v6; // bx
  __int64 v7; // r8
  int v8; // eax
  __int64 v9; // r15
  int v10; // r9d
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // rsi
  unsigned __int8 *result; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rsi
  _QWORD *v20; // r9
  __m128i v21; // xmm0
  unsigned int v22; // esi
  unsigned __int8 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // r15
  __int128 v26; // [rsp-10h] [rbp-E0h]
  __int64 v27; // [rsp+10h] [rbp-C0h]
  __int64 v28; // [rsp+10h] [rbp-C0h]
  _QWORD *v29; // [rsp+10h] [rbp-C0h]
  __int16 v30; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v31; // [rsp+18h] [rbp-B8h]
  __int16 v32; // [rsp+18h] [rbp-B8h]
  __int64 v33; // [rsp+40h] [rbp-90h] BYREF
  int v34; // [rsp+48h] [rbp-88h]
  __int64 v35; // [rsp+50h] [rbp-80h] BYREF
  int v36; // [rsp+58h] [rbp-78h]
  unsigned __int16 v37; // [rsp+60h] [rbp-70h] BYREF
  __int64 v38; // [rsp+68h] [rbp-68h]
  __int16 v39; // [rsp+70h] [rbp-60h]
  __int64 v40; // [rsp+78h] [rbp-58h]
  __m128i v41; // [rsp+80h] [rbp-50h] BYREF
  __int64 v42; // [rsp+90h] [rbp-40h]
  unsigned __int64 v43; // [rsp+98h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 48);
  v5 = *(unsigned __int64 **)(a2 + 40);
  v6 = *(_WORD *)v4;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 > 239 )
  {
    if ( (unsigned int)(v8 - 242) > 1 )
      goto LABEL_4;
  }
  else if ( v8 <= 237 && (unsigned int)(v8 - 101) > 0x2F )
  {
LABEL_4:
    v27 = v7;
    v9 = v5[1];
    v30 = *(_WORD *)(*(_QWORD *)(*v5 + 48) + 16LL * (unsigned int)v9);
    sub_380F170(a1, *v5, v9);
    v11 = v27;
    v12 = *(_QWORD *)(a1 + 8);
    v35 = *(_QWORD *)(a2 + 80);
    if ( v35 )
    {
      sub_B96E90((__int64)&v35, v35, 1);
      v11 = v27;
    }
    v36 = *(_DWORD *)(a2 + 72);
    if ( v30 == 11 )
    {
      v13 = 236;
      goto LABEL_8;
    }
    if ( v6 == 11 )
    {
      v13 = 237;
    }
    else if ( v30 == 10 )
    {
      v13 = 240;
    }
    else
    {
      v13 = 241;
      if ( v6 != 10 )
        goto LABEL_22;
    }
LABEL_8:
    result = sub_33FAF80(v12, v13, (__int64)&v35, v6, v11, v10, a3);
    if ( v35 )
    {
      v31 = result;
      sub_B91220((__int64)&v35, v35);
      return v31;
    }
    return result;
  }
  v15 = v5[5];
  v16 = v5[6];
  v28 = v7;
  v32 = *(_WORD *)(*(_QWORD *)(v15 + 48) + 16LL * (unsigned int)v16);
  v17 = sub_380F170(a1, v15, v16);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_QWORD **)(a1 + 8);
  v21 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v43 = v18 | v16 & 0xFFFFFFFF00000000LL;
  v42 = v17;
  v37 = v6;
  v38 = v28;
  v39 = 1;
  v40 = 0;
  v33 = v19;
  v41 = v21;
  if ( v19 )
  {
    v29 = v20;
    sub_B96E90((__int64)&v33, v19, 1);
    v20 = v29;
  }
  v34 = *(_DWORD *)(a2 + 72);
  if ( v32 == 11 )
  {
    v22 = 238;
  }
  else if ( v6 == 11 )
  {
    v22 = 239;
  }
  else if ( v32 == 10 )
  {
    v22 = 242;
  }
  else
  {
    if ( v6 != 10 )
LABEL_22:
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v22 = 243;
  }
  *((_QWORD *)&v26 + 1) = 2;
  *(_QWORD *)&v26 = &v41;
  v23 = sub_3411BE0(v20, v22, (__int64)&v33, &v37, 2, (__int64)v20, v26);
  v25 = v24;
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v23, 1);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v23, v25);
  return 0;
}
