// Function: sub_26F2E10
// Address: 0x26f2e10
//
__int64 __fastcall sub_26F2E10(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 i; // r12
  __int64 v6; // rax
  __int64 j; // r12
  __int64 v8; // rax
  __int64 k; // r12
  __int64 v10; // rax
  __int64 m; // r12
  __m128i v12; // xmm1
  __int64 *v13; // r13
  __int64 *v14; // r15
  __m128i *v15; // rdi
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 *v19; // r13
  __int64 *v20; // r15
  __int64 *v21; // rdi
  __int64 v22; // r8
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r10
  __int64 v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r9
  unsigned __int8 v30; // al
  int v31; // edx
  __m128i *v32; // rbx
  __int64 (__fastcall *v33)(__int64); // rax
  __m128i *v34; // r15
  __m128i *v35; // rdi
  _QWORD *v36; // rdi
  int v38; // [rsp+Ch] [rbp-164h]
  __m128i v39; // [rsp+10h] [rbp-160h]
  __m128i v40; // [rsp+20h] [rbp-150h]
  _QWORD v41[2]; // [rsp+50h] [rbp-120h] BYREF
  _QWORD v42[4]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v43; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+88h] [rbp-E8h]
  __int64 v45; // [rsp+90h] [rbp-E0h]
  unsigned int v46; // [rsp+98h] [rbp-D8h]
  __m128i v47; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v48; // [rsp+B0h] [rbp-C0h]
  __int64 v49[4]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 *v50; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v51; // [rsp+E8h] [rbp-88h]
  _QWORD v52[2]; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v53; // [rsp+100h] [rbp-70h] BYREF
  __m128i v54[6]; // [rsp+110h] [rbp-60h] BYREF

  v41[0] = a2;
  v42[0] = &v43;
  v41[1] = a3;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v42[1] = v41;
  v42[2] = a1;
  v4 = sub_B6AC80((__int64)a1, 358);
  if ( v4 )
  {
    for ( i = *(_QWORD *)(v4 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_26F2A90((__int64)v42, *(_QWORD *)(i + 24), 1u);
  }
  v6 = sub_B6AC80((__int64)a1, 300);
  if ( v6 )
  {
    for ( j = *(_QWORD *)(v6 + 16); j; j = *(_QWORD *)(j + 8) )
      sub_26F2A90((__int64)v42, *(_QWORD *)(j + 24), 1u);
  }
  v8 = sub_B6AC80((__int64)a1, 356);
  if ( v8 )
  {
    for ( k = *(_QWORD *)(v8 + 16); k; k = *(_QWORD *)(k + 8) )
      sub_26F2A90((__int64)v42, *(_QWORD *)(k + 24), 2u);
  }
  v10 = sub_B6AC80((__int64)a1, 357);
  if ( v10 )
  {
    for ( m = *(_QWORD *)(v10 + 16); m; m = *(_QWORD *)(m + 8) )
      sub_26F2A90((__int64)v42, *(_QWORD *)(m + 24), 2u);
  }
  sub_BA9600(&v53, (__int64)a1);
  v12 = _mm_loadu_si128(v54);
  v40 = v54[1];
  v47 = _mm_loadu_si128(&v53);
  v48 = v12;
  v39 = v54[2];
  while ( *(_OWORD *)&v47 != *(_OWORD *)&v40 || *(_OWORD *)&v39 != *(_OWORD *)&v48 )
  {
    v13 = v49;
    v14 = v49;
    v15 = &v47;
    v49[2] = (__int64)sub_25AC5E0;
    v16 = sub_25AC5C0;
    v49[3] = 0;
    if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
      goto LABEL_17;
    while ( 1 )
    {
      v16 = *(__int64 (__fastcall **)(__int64))((char *)v16 + v15->m128i_i64[0] - 1);
LABEL_17:
      v17 = v16((__int64)v15);
      if ( v17 )
        break;
      while ( 1 )
      {
        v13 += 2;
        if ( &v50 == (__int64 **)v13 )
LABEL_46:
          BUG();
        v16 = (__int64 (__fastcall *)(__int64))v14[2];
        v18 = v14[3];
        v14 = v13;
        v15 = (__m128i *)((char *)&v47 + v18);
        if ( ((unsigned __int8)v16 & 1) != 0 )
          break;
        v17 = v16((__int64)v15);
        if ( v17 )
          goto LABEL_21;
      }
    }
LABEL_21:
    v50 = v52;
    v51 = 0x100000000LL;
    sub_B91D10(v17, 19, (__int64)&v50);
    sub_B98000(v17, 19);
    v19 = v50;
    v20 = &v50[(unsigned int)v51];
    if ( v20 != v50 )
    {
      while ( 1 )
      {
        v29 = *v19;
        v30 = *(_BYTE *)(*v19 - 16);
        if ( (v30 & 2) != 0 )
          v21 = *(__int64 **)(v29 - 32);
        else
          v21 = (__int64 *)(v29 + -16 - 8LL * ((v30 >> 2) & 0xF));
        v22 = v21[1];
        if ( !v46 )
          goto LABEL_32;
        v23 = (v46 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v24 = (__int64 *)(v44 + 16LL * v23);
        v25 = *v24;
        if ( v22 != *v24 )
        {
          v31 = 1;
          while ( v25 != -4096 )
          {
            v23 = (v46 - 1) & (v31 + v23);
            v38 = v31 + 1;
            v24 = (__int64 *)(v44 + 16LL * v23);
            v25 = *v24;
            if ( v22 == *v24 )
              goto LABEL_26;
            v31 = v38;
          }
          goto LABEL_32;
        }
LABEL_26:
        if ( v24 == (__int64 *)(v44 + 16LL * v46) )
        {
LABEL_32:
          ++v19;
          sub_B994D0(v17, 19, v29);
          if ( v20 == v19 )
          {
LABEL_33:
            v19 = v50;
            break;
          }
        }
        else
        {
          v26 = *v21;
          v27 = *a1;
          ++v19;
          v49[0] = v26;
          v49[1] = v24[1];
          v28 = sub_B9C770(v27, v49, (__int64 *)2, 0, 1);
          sub_B994D0(v17, 19, v28);
          if ( v20 == v19 )
            goto LABEL_33;
        }
      }
    }
    if ( v19 != v52 )
      _libc_free((unsigned __int64)v19);
    v32 = (__m128i *)&v50;
    v52[1] = 0;
    v52[0] = sub_25AC590;
    v33 = sub_25AC560;
    v34 = (__m128i *)&v50;
    v35 = &v47;
    if ( ((unsigned __int8)sub_25AC560 & 1) != 0 )
LABEL_37:
      v33 = *(__int64 (__fastcall **)(__int64))((char *)v33 + v35->m128i_i64[0] - 1);
    while ( !(unsigned __int8)v33((__int64)v35) )
    {
      if ( &v53 == ++v32 )
        goto LABEL_46;
      v36 = (_QWORD *)v34[1].m128i_i64[1];
      v33 = (__int64 (__fastcall *)(__int64))v34[1].m128i_i64[0];
      v34 = v32;
      v35 = (__m128i *)((char *)&v47 + (_QWORD)v36);
      if ( ((unsigned __int8)v33 & 1) != 0 )
        goto LABEL_37;
    }
  }
  return sub_C7D6A0(v44, 16LL * v46, 8);
}
