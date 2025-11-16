// Function: sub_33AD3D0
// Address: 0x33ad3d0
//
void __fastcall sub_33AD3D0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rsi
  __int128 v7; // rax
  __int64 v8; // r14
  int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  __int16 v13; // ax
  unsigned __int64 v14; // rdx
  _OWORD *v15; // rax
  _OWORD *v16; // rcx
  unsigned int v17; // r13d
  unsigned __int64 v18; // r15
  __int64 v19; // r14
  __int128 v20; // rax
  int v21; // r9d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // r13
  __int64 v28; // r14
  int v29; // eax
  int v30; // edx
  int v31; // r9d
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // r14
  _QWORD *v35; // rax
  _OWORD *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r13
  __m128i v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // r9
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rsi
  __m128i v48; // xmm1
  __int64 v49; // r13
  __int64 v50; // rdx
  __int64 v51; // r15
  _QWORD *v52; // rax
  __int128 v53; // [rsp-10h] [rbp-1F0h]
  __int64 v56; // [rsp+18h] [rbp-1C8h]
  __m128i v57; // [rsp+20h] [rbp-1C0h] BYREF
  __int128 v58; // [rsp+30h] [rbp-1B0h]
  __int64 v59; // [rsp+40h] [rbp-1A0h]
  __int64 v60; // [rsp+48h] [rbp-198h]
  __int64 v61; // [rsp+50h] [rbp-190h]
  __int64 v62; // [rsp+58h] [rbp-188h]
  __int64 v63; // [rsp+60h] [rbp-180h]
  __int64 v64; // [rsp+68h] [rbp-178h]
  __int64 v65; // [rsp+70h] [rbp-170h]
  __int64 v66; // [rsp+78h] [rbp-168h]
  __int64 v67; // [rsp+88h] [rbp-158h]
  __int64 v68; // [rsp+90h] [rbp-150h] BYREF
  int v69; // [rsp+98h] [rbp-148h]
  __m128i v70; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-130h] BYREF
  int v72; // [rsp+B8h] [rbp-128h]
  const __m128i *v73[2]; // [rsp+C0h] [rbp-120h] BYREF
  _BYTE v74[64]; // [rsp+D0h] [rbp-110h] BYREF
  _QWORD *v75; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+118h] [rbp-C8h]
  _OWORD v77[4]; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v78; // [rsp+160h] [rbp-80h] BYREF
  _QWORD v79[14]; // [rsp+170h] [rbp-70h] BYREF

  v4 = *(_QWORD *)a1;
  v5 = *(_DWORD *)(a1 + 848);
  v68 = 0;
  v69 = v5;
  if ( v4 )
  {
    if ( &v68 != (__int64 *)(v4 + 48) )
    {
      v6 = *(_QWORD *)(v4 + 48);
      v68 = v6;
      if ( v6 )
        sub_B96E90((__int64)&v68, v6, 1);
    }
  }
  *(_QWORD *)&v7 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v8 = *(_QWORD *)(a2 + 8);
  v73[1] = (const __m128i *)0x400000000LL;
  v58 = v7;
  v73[0] = (const __m128i *)v74;
  v9 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v78.m128i_i64[0] = 0;
  v78.m128i_i8[8] = 0;
  sub_34B8C80(v10, v9, v8, (unsigned int)v73, 0, 0, __PAIR128__(v78.m128i_u64[1], 0));
  v70 = _mm_loadu_si128(v73[0]);
  v13 = v70.m128i_i16[0];
  if ( v70.m128i_i16[0] )
  {
    v76 = 0x400000000LL;
    v14 = a3;
    LODWORD(v59) = word_4456340[v70.m128i_u16[0] - 1];
    v75 = v77;
    if ( !a3 )
      goto LABEL_15;
  }
  else
  {
    v57.m128i_i64[0] = (__int64)&v70;
    v37 = sub_3007240((__int64)&v70);
    v14 = a3;
    v76 = 0x400000000LL;
    v67 = v37;
    LODWORD(v59) = v37;
    v75 = v77;
    if ( !a3 )
      goto LABEL_25;
  }
  v15 = v77;
  if ( v14 > 4 )
  {
    v57.m128i_i64[0] = v14;
    sub_C8D5F0((__int64)&v75, v77, v14, 0x10u, v11, v12);
    v14 = v57.m128i_i64[0];
    v15 = &v75[2 * (unsigned int)v76];
    v16 = &v75[2 * v57.m128i_i64[0]];
    if ( v16 == v15 )
      goto LABEL_12;
  }
  else
  {
    v16 = &v77[v14];
    if ( v16 == v77 )
      goto LABEL_12;
  }
  do
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = 0;
      *((_DWORD *)v15 + 2) = 0;
    }
    ++v15;
  }
  while ( v16 != v15 );
LABEL_12:
  v57.m128i_i64[0] = 16 * v14;
  LODWORD(v76) = a3;
  v17 = 0;
  v18 = 0;
  do
  {
    v19 = *(_QWORD *)(a1 + 864);
    *(_QWORD *)&v20 = sub_3400EE0(v19, v17, &v68, 0, v11);
    v22 = sub_3406EB0(v19, 161, (unsigned int)&v68, v70.m128i_i32[0], v70.m128i_i32[2], v21, v58, v20);
    v17 += v59;
    v24 = v23;
    v25 = v22;
    v26 = v75;
    v65 = v25;
    v66 = v24;
    v75[v18 / 8] = v25;
    LODWORD(v26[v18 / 8 + 1]) = v66;
    v18 += 16LL;
  }
  while ( v57.m128i_i64[0] != v18 );
  v13 = v70.m128i_i16[0];
  if ( v70.m128i_i16[0] )
  {
LABEL_15:
    if ( (unsigned __int16)(v13 - 17) > 0x9Eu )
      goto LABEL_17;
    goto LABEL_16;
  }
LABEL_25:
  if ( !sub_30070D0((__int64)&v70) )
  {
LABEL_17:
    v27 = v75;
    v28 = (unsigned int)v76;
    v59 = *(_QWORD *)(a1 + 864);
    v29 = sub_33E5830(v59, v73[0]);
    *((_QWORD *)&v53 + 1) = v28;
    *(_QWORD *)&v53 = v27;
    v32 = sub_3411630(v59, 162, (unsigned int)&v68, v29, v30, v31, v53);
    v34 = v33;
    v78.m128i_i64[0] = a2;
    v35 = sub_337DC20(a1 + 8, v78.m128i_i64);
    v64 = v34;
    v63 = v32;
    *v35 = v32;
    v36 = v75;
    *((_DWORD *)v35 + 2) = v64;
    if ( v36 == v77 )
      goto LABEL_19;
    goto LABEL_18;
  }
LABEL_16:
  if ( a3 != 2 )
    goto LABEL_17;
  v38 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v58 = &v78;
  sub_9B95E0(v78.m128i_i64, 0, 2, v59);
  v39.m128i_i64[0] = sub_33FCE10(
                       v38,
                       v70.m128i_i32[0],
                       v70.m128i_i32[2],
                       (unsigned int)&v68,
                       *v75,
                       v75[1],
                       v75[2],
                       v75[3],
                       v78.m128i_i64[0],
                       v78.m128i_u32[2]);
  v57 = v39;
  if ( (_QWORD *)v78.m128i_i64[0] != v79 )
    _libc_free(v78.m128i_u64[0]);
  v56 = *(_QWORD *)(a1 + 864);
  sub_9B95E0((__int64 *)v58, 1, 2, v59);
  v40 = sub_33FCE10(
          v56,
          v70.m128i_i32[0],
          v70.m128i_i32[2],
          (unsigned int)&v68,
          *v75,
          v75[1],
          v75[2],
          v75[3],
          v78.m128i_i64[0],
          v78.m128i_u32[2]);
  v42 = v40;
  v43 = v41;
  if ( (_QWORD *)v78.m128i_i64[0] != v79 )
  {
    v59 = v40;
    v60 = v41;
    _libc_free(v78.m128i_u64[0]);
    v42 = v59;
    v43 = v60;
  }
  v44 = *(_DWORD *)(a1 + 848);
  v45 = *(_QWORD *)a1;
  v71 = 0;
  v46 = *(_QWORD *)(a1 + 864);
  v72 = v44;
  if ( v45 )
  {
    if ( &v71 != (__int64 *)(v45 + 48) )
    {
      v47 = *(_QWORD *)(v45 + 48);
      v71 = v47;
      if ( v47 )
      {
        v59 = v42;
        v60 = v43;
        sub_B96E90((__int64)&v71, v47, 1);
        v42 = v59;
        v43 = v60;
      }
    }
  }
  v79[0] = v42;
  v48 = _mm_load_si128(&v57);
  v79[1] = v43;
  v78 = v48;
  v49 = sub_3411660(v46, v58, 2, &v71);
  v51 = v50;
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
  v78.m128i_i64[0] = a2;
  v52 = sub_337DC20(a1 + 8, (__int64 *)v58);
  v62 = v51;
  v61 = v49;
  *v52 = v49;
  v36 = v75;
  *((_DWORD *)v52 + 2) = v62;
  if ( v36 != v77 )
LABEL_18:
    _libc_free((unsigned __int64)v36);
LABEL_19:
  if ( v73[0] != (const __m128i *)v74 )
    _libc_free((unsigned __int64)v73[0]);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
}
