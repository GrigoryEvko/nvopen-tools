// Function: sub_203C550
// Address: 0x203c550
//
__int64 *__fastcall sub_203C550(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  __int64 **v6; // r13
  char *v8; // rax
  char v9; // dl
  const void **v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rsi
  const void **v13; // rdx
  __m128i v14; // xmm0
  unsigned int v15; // edx
  unsigned int v16; // r15d
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  char v19; // al
  __int64 v20; // rdx
  int v21; // ecx
  _QWORD *v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rcx
  const void **v26; // r8
  __int64 v27; // rax
  __int64 *v28; // r13
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r15
  __int64 v31; // r14
  __int128 v32; // rax
  __int64 *v33; // r14
  const void **v35; // rdx
  unsigned int v36; // eax
  int v37; // r8d
  int v38; // r9d
  const void **v39; // rdx
  __int64 v40; // rbx
  _BYTE *v41; // rdx
  _BYTE *v42; // rax
  __int64 v43; // rbx
  unsigned int v44; // r13d
  char v45; // al
  __int128 v46; // rax
  __int128 v47; // rax
  __int64 v48; // rax
  int v49; // edx
  int v50; // edi
  __int64 v51; // rdx
  __int64 v52; // rax
  _BYTE *v53; // rax
  __int64 *v54; // r14
  __int64 (__fastcall *v55)(__int64, __int64); // r15
  __int64 v56; // rax
  unsigned int v57; // edx
  __int128 v58; // [rsp-10h] [rbp-1F0h]
  __int64 v59; // [rsp+8h] [rbp-1D8h]
  unsigned int v60; // [rsp+10h] [rbp-1D0h]
  const void **v61; // [rsp+18h] [rbp-1C8h]
  __int64 v62; // [rsp+20h] [rbp-1C0h]
  const void **v64; // [rsp+30h] [rbp-1B0h]
  unsigned int v65; // [rsp+38h] [rbp-1A8h]
  unsigned int v66; // [rsp+40h] [rbp-1A0h]
  unsigned int v67; // [rsp+44h] [rbp-19Ch]
  unsigned int v68; // [rsp+48h] [rbp-198h]
  __int64 *v69; // [rsp+48h] [rbp-198h]
  __int128 v70; // [rsp+50h] [rbp-190h]
  __int64 v71; // [rsp+70h] [rbp-170h] BYREF
  const void **v72; // [rsp+78h] [rbp-168h]
  __int64 v73; // [rsp+80h] [rbp-160h] BYREF
  int v74; // [rsp+88h] [rbp-158h]
  _BYTE v75[8]; // [rsp+90h] [rbp-150h] BYREF
  __int64 v76; // [rsp+98h] [rbp-148h]
  _BYTE *v77; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v78; // [rsp+A8h] [rbp-138h]
  _BYTE v79[304]; // [rsp+B0h] [rbp-130h] BYREF

  v6 = a1;
  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v10 = (const void **)*((_QWORD *)v8 + 1);
  LOBYTE(v71) = v9;
  v72 = v10;
  LOBYTE(v11) = sub_1F7E0F0((__int64)&v71);
  v12 = *(_QWORD *)(a2 + 72);
  v64 = v13;
  v65 = v11;
  v73 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v73, v12, 2);
  v74 = *(_DWORD *)(a2 + 64);
  if ( (_BYTE)v71 )
    v67 = word_4305480[(unsigned __int8)(v71 - 14)];
  else
    v67 = sub_1F58D30((__int64)&v71);
  v14 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v62 = sub_20363F0((__int64)a1, v14.m128i_u64[0], v14.m128i_i64[1]);
  v16 = v15;
  *(_QWORD *)&v70 = v62;
  v17 = v15 | v14.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v18 = *(_QWORD *)(v62 + 40) + 16LL * v15;
  *((_QWORD *)&v70 + 1) = v17;
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v66 = *(unsigned __int16 *)(a2 + 24);
  v75[0] = v19;
  v76 = v20;
  if ( v19 )
    v21 = word_4305480[(unsigned __int8)(v19 - 14)];
  else
    v21 = sub_1F58D30((__int64)v75);
  v68 = v21;
  v22 = (_QWORD *)a1[1][6];
  LOBYTE(v23) = sub_1D15020(v65, v21);
  if ( (_BYTE)v23 )
  {
    v24 = *a1;
    v25 = (unsigned __int8)v23;
    v26 = 0;
  }
  else
  {
    v23 = sub_1F593D0(v22, v65, (__int64)v64, v68);
    v25 = v23;
    v26 = v35;
    if ( !(_BYTE)v23 )
      goto LABEL_15;
    v24 = *a1;
  }
  if ( v24[(unsigned __int8)v23 + 15] )
  {
    LOBYTE(v25) = v23;
    v27 = sub_1D309E0(
            a1[1],
            v66,
            (__int64)&v73,
            v25,
            v26,
            0,
            *(double *)v14.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v70);
    v28 = a1[1];
    v30 = v29;
    v31 = v27;
    *(_QWORD *)&v32 = sub_1D38E70((__int64)v28, 0, (__int64)&v73, 0, v14, a4, a5);
    v33 = sub_1D332F0(
            v28,
            109,
            (__int64)&v73,
            (unsigned int)v71,
            v72,
            0,
            *(double *)v14.m128i_i64,
            a4,
            a5,
            v31,
            v30,
            v32);
    goto LABEL_11;
  }
LABEL_15:
  LOBYTE(v36) = sub_1F7E0F0((__int64)v75);
  v61 = v39;
  v60 = v36;
  v77 = v79;
  v40 = 16LL * v67;
  v78 = 0x1000000000LL;
  if ( v67 > 0x10uLL )
  {
    sub_16CD150((__int64)&v77, v79, v67, 16, v37, v38);
    LODWORD(v78) = v67;
    v42 = v77;
    v41 = &v77[v40];
    if ( v77 == &v77[v40] )
    {
LABEL_21:
      v43 = 0;
      v44 = v5;
      v59 = v16;
      do
      {
        v54 = a1[1];
        v69 = *a1;
        v55 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
        v56 = sub_1E0A0C0(v54[4]);
        if ( v55 == sub_1D13A20 )
        {
          v57 = 8 * sub_15A9520(v56, 0);
          if ( v57 == 32 )
          {
            v45 = 5;
          }
          else if ( v57 <= 0x20 )
          {
            v45 = 3;
            if ( v57 != 8 )
              v45 = 4 * (v57 == 16);
          }
          else
          {
            v45 = 6;
            if ( v57 != 64 )
            {
              v45 = 0;
              if ( v57 == 128 )
                v45 = 7;
            }
          }
        }
        else
        {
          v45 = v55((__int64)v69, v56);
        }
        LOBYTE(v44) = v45;
        *(_QWORD *)&v46 = sub_1D38BB0((__int64)v54, v43, (__int64)&v73, v44, 0, 0, v14, a4, a5, 0);
        *((_QWORD *)&v70 + 1) = v59 | *((_QWORD *)&v70 + 1) & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v47 = sub_1D332F0(
                            v54,
                            106,
                            (__int64)&v73,
                            v60,
                            v61,
                            0,
                            *(double *)v14.m128i_i64,
                            a4,
                            a5,
                            v62,
                            *((unsigned __int64 *)&v70 + 1),
                            v46);
        v48 = sub_1D309E0(
                v54,
                v66,
                (__int64)&v73,
                v65,
                v64,
                0,
                *(double *)v14.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                v47);
        v50 = v49;
        v51 = v48;
        v52 = v43++;
        v53 = &v77[16 * v52];
        *(_QWORD *)v53 = v51;
        *((_DWORD *)v53 + 2) = v50;
      }
      while ( v67 > (unsigned int)v43 );
      v6 = a1;
      goto LABEL_38;
    }
  }
  else
  {
    v41 = &v79[v40];
    LODWORD(v78) = v67;
    v42 = v79;
    if ( &v79[v40] == v79 )
      goto LABEL_20;
  }
  do
  {
    if ( v42 )
    {
      *(_QWORD *)v42 = 0;
      *((_DWORD *)v42 + 2) = 0;
    }
    v42 += 16;
  }
  while ( v42 != v41 );
LABEL_20:
  if ( v67 )
    goto LABEL_21;
LABEL_38:
  *((_QWORD *)&v58 + 1) = (unsigned int)v78;
  *(_QWORD *)&v58 = v77;
  v33 = sub_1D359D0(v6[1], 104, (__int64)&v73, v71, v72, 0, *(double *)v14.m128i_i64, a4, a5, v58);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
LABEL_11:
  if ( v73 )
    sub_161E7C0((__int64)&v73, v73);
  return v33;
}
