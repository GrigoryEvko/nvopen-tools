// Function: sub_2039C70
// Address: 0x2039c70
//
__int64 *__fastcall sub_2039C70(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  int v5; // r14d
  char *v7; // rax
  __int64 v8; // rsi
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int128 v13; // xmm1
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // r15
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rdx
  __int64 v20; // r15
  bool v21; // cc
  char v22; // al
  const void **v23; // rcx
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r15
  _QWORD *v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // r9
  const void **v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 *v34; // r12
  unsigned __int64 v35; // r14
  char v36; // al
  __int64 v37; // rcx
  __int128 v38; // rax
  __int64 *v39; // rax
  int v40; // edx
  int v41; // r11d
  __int64 *v42; // rdx
  unsigned __int64 v43; // rax
  __int64 *v44; // r15
  __int64 v45; // rax
  unsigned int v46; // edx
  __int64 *v47; // r12
  _QWORD *v49; // rbx
  int v50; // edx
  int v51; // r12d
  __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  unsigned int v54; // edx
  __int128 v55; // [rsp-10h] [rbp-230h]
  unsigned int v56; // [rsp+8h] [rbp-218h]
  unsigned int v57; // [rsp+Ch] [rbp-214h]
  __int64 v58; // [rsp+10h] [rbp-210h]
  unsigned int v59; // [rsp+28h] [rbp-1F8h]
  const void **v60; // [rsp+30h] [rbp-1F0h]
  __int64 v61; // [rsp+38h] [rbp-1E8h]
  __int64 v62; // [rsp+40h] [rbp-1E0h]
  __int64 (__fastcall *v63)(__int64, __int64); // [rsp+50h] [rbp-1D0h]
  unsigned int v64; // [rsp+68h] [rbp-1B8h]
  __int64 v65; // [rsp+68h] [rbp-1B8h]
  __int64 v66; // [rsp+90h] [rbp-190h] BYREF
  __int64 v67; // [rsp+98h] [rbp-188h]
  __int64 v68; // [rsp+A0h] [rbp-180h] BYREF
  const void **v69; // [rsp+A8h] [rbp-178h]
  __int64 v70; // [rsp+B0h] [rbp-170h] BYREF
  int v71; // [rsp+B8h] [rbp-168h]
  char v72[8]; // [rsp+C0h] [rbp-160h] BYREF
  const void **v73; // [rsp+C8h] [rbp-158h]
  __int64 v74; // [rsp+D0h] [rbp-150h] BYREF
  int v75; // [rsp+D8h] [rbp-148h]
  _QWORD *v76; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v77; // [rsp+E8h] [rbp-138h]
  _QWORD v78[38]; // [rsp+F0h] [rbp-130h] BYREF

  v7 = *(char **)(a2 + 40);
  v8 = *(_QWORD *)a1;
  v9 = *v7;
  v67 = *((_QWORD *)v7 + 1);
  v10 = *(_QWORD *)(a1 + 8);
  LOBYTE(v66) = v9;
  sub_1F40D10((__int64)&v76, v8, *(_QWORD *)(v10 + 48), v66, v67);
  LOBYTE(v68) = v77;
  v69 = (const void **)v78[0];
  if ( (_BYTE)v77 )
    v57 = word_4305480[(unsigned __int8)(v77 - 14)];
  else
    v57 = sub_1F58D30((__int64)&v68);
  v11 = *(_QWORD *)(a2 + 32);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = (__int128)_mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(_QWORD *)(v11 + 40);
  v61 = *(_QWORD *)v11;
  v64 = *(_DWORD *)(v11 + 8);
  v15 = *(_QWORD *)(a2 + 72);
  v70 = v15;
  if ( v15 )
    sub_1623A60((__int64)&v70, v15, 2);
  v71 = *(_DWORD *)(a2 + 64);
  v16 = 16LL * v64;
  sub_1F40D10(
    (__int64)&v76,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    *(unsigned __int8 *)(v16 + *(_QWORD *)(v61 + 40)),
    *(_QWORD *)(v16 + *(_QWORD *)(v61 + 40) + 8));
  if ( (_BYTE)v76 == 7 )
  {
    v61 = sub_20363F0(a1, v12.m128i_u64[0], v12.m128i_i64[1]);
    v64 = v54;
    v16 = 16LL * v54;
  }
  v19 = *(_QWORD *)(v14 + 88);
  v20 = *(_QWORD *)(v61 + 40) + v16;
  v21 = *(_DWORD *)(v19 + 32) <= 0x40u;
  v22 = *(_BYTE *)v20;
  v23 = *(const void ***)(v20 + 8);
  v24 = *(_QWORD *)(v19 + 24);
  v72[0] = *(_BYTE *)v20;
  v73 = v23;
  if ( !v21 )
    v24 = *(_QWORD *)v24;
  if ( !v24 && (_BYTE)v68 == v22 )
  {
    if ( v69 == v23 || v22 )
    {
      v47 = (__int64 *)v61;
      goto LABEL_40;
    }
    goto LABEL_38;
  }
  if ( !v22 )
  {
LABEL_38:
    v25 = (unsigned int)sub_1F58D30((__int64)v72);
    goto LABEL_13;
  }
  v25 = word_4305480[(unsigned __int8)(v22 - 14)];
LABEL_13:
  if ( v24 % v57 || v57 + v24 >= v25 )
  {
    v26 = v78;
    v76 = v78;
    v77 = 0x1000000000LL;
    if ( v57 > 0x10uLL )
    {
      sub_16CD150((__int64)&v76, v78, v57, 16, v17, v18);
      v26 = v76;
    }
    v27 = 2LL * v57;
    v28 = &v26[v27];
    for ( LODWORD(v77) = v57; v28 != v26; v26 += 2 )
    {
      if ( v26 )
      {
        *v26 = 0;
        *((_DWORD *)v26 + 2) = 0;
      }
    }
    LOBYTE(v29) = sub_1F7E0F0((__int64)&v66);
    v59 = v29;
    v60 = v31;
    if ( (_BYTE)v66 )
      v56 = word_4305480[(unsigned __int8)(v66 - 14)];
    else
      v56 = sub_1F58D30((__int64)&v66);
    if ( v56 )
    {
      v32 = 0;
      v33 = v64;
      v65 = v24;
      LODWORD(v34) = v5;
      v35 = v12.m128i_u64[1];
      v58 = v33;
      do
      {
        v44 = *(__int64 **)(a1 + 8);
        v62 = *(_QWORD *)a1;
        v63 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
        v45 = sub_1E0A0C0(v44[4]);
        if ( v63 == sub_1D13A20 )
        {
          v46 = 8 * sub_15A9520(v45, 0);
          if ( v46 == 32 )
          {
            v36 = 5;
          }
          else if ( v46 <= 0x20 )
          {
            v36 = 3;
            if ( v46 != 8 )
              v36 = 4 * (v46 == 16);
          }
          else
          {
            v36 = 6;
            if ( v46 != 64 )
            {
              v36 = 0;
              if ( v46 == 128 )
                v36 = 7;
            }
          }
        }
        else
        {
          v36 = v63(v62, v45);
        }
        LOBYTE(v34) = v36;
        v37 = (unsigned int)v34;
        v34 = &v70;
        *(_QWORD *)&v38 = sub_1D38BB0((__int64)v44, v65, (__int64)&v70, v37, 0, 0, v12, *(double *)&v13, a5, 0);
        v35 = v35 & 0xFFFFFFFF00000000LL | v58;
        v39 = sub_1D332F0(
                v44,
                106,
                (__int64)&v70,
                v59,
                v60,
                0,
                *(double *)v12.m128i_i64,
                *(double *)&v13,
                a5,
                v61,
                v35,
                v38);
        ++v65;
        v41 = v40;
        v42 = v39;
        v43 = (unsigned __int64)v76;
        v76[v32] = v42;
        *(_DWORD *)(v43 + v32 * 8 + 8) = v41;
        v32 += 2;
      }
      while ( 2LL * v56 != v32 );
    }
    v74 = 0;
    v75 = 0;
    v49 = sub_1D2B300(*(_QWORD **)(a1 + 8), 0x30u, (__int64)&v74, v59, (__int64)v60, v30);
    v51 = v50;
    if ( v74 )
      sub_161E7C0((__int64)&v74, v74);
    if ( v57 > v56 )
    {
      v52 = 2LL * v56;
      do
      {
        v53 = (unsigned __int64)v76;
        v76[v52] = v49;
        *(_DWORD *)(v53 + v52 * 8 + 8) = v51;
        v52 += 2;
      }
      while ( v52 != 2 * (v56 + (unsigned __int64)(v57 - 1 - v56) + 1) );
    }
    *((_QWORD *)&v55 + 1) = (unsigned int)v77;
    *(_QWORD *)&v55 = v76;
    v47 = sub_1D359D0(
            *(__int64 **)(a1 + 8),
            104,
            (__int64)&v70,
            v68,
            v69,
            0,
            *(double *)v12.m128i_i64,
            *(double *)&v13,
            a5,
            v55);
    if ( v76 != v78 )
      _libc_free((unsigned __int64)v76);
  }
  else
  {
    v47 = sub_1D332F0(
            *(__int64 **)(a1 + 8),
            109,
            (__int64)&v70,
            (unsigned int)v68,
            v69,
            0,
            *(double *)v12.m128i_i64,
            *(double *)&v13,
            a5,
            v61,
            v64 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL,
            v13);
  }
LABEL_40:
  if ( v70 )
    sub_161E7C0((__int64)&v70, v70);
  return v47;
}
