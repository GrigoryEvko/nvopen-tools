// Function: sub_1FD24F0
// Address: 0x1fd24f0
//
__int64 __fastcall sub_1FD24F0(__int64 a1, __int64 a2, unsigned int a3, __m128 a4, __m128 a5, __m128i a6)
{
  char v9; // bl
  __int64 v10; // r9
  char *v11; // rax
  unsigned __int8 v12; // dl
  const void **v13; // rbx
  __int64 *v14; // r14
  __int64 v15; // r10
  unsigned __int64 v16; // rcx
  __int64 v17; // r11
  __int64 v18; // rsi
  __int64 *v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r14
  unsigned int v22; // ebx
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // rsi
  __int64 *v28; // r14
  __int64 v29; // r10
  __int64 v30; // rdx
  const void **v31; // rbx
  unsigned __int64 v32; // rcx
  __int64 v33; // r11
  __int64 *v34; // rbx
  __int128 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 *v40; // rax
  unsigned int v41; // edx
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned __int8 v46; // al
  const void **v47; // r8
  __int64 *v48; // rbx
  __int64 v49; // r10
  unsigned __int64 v50; // rcx
  __int64 v51; // r11
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 v54; // rcx
  const void **v55; // r8
  __int64 *v56; // rbx
  __int64 v57; // r10
  unsigned __int64 v58; // rcx
  __int64 v59; // r11
  __int64 v60; // rsi
  __int64 *v61; // rbx
  int v62; // edx
  int v63; // r14d
  __int128 v64; // rdi
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 *v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rdi
  __int128 v74; // [rsp-10h] [rbp-90h]
  __int128 v75; // [rsp-10h] [rbp-90h]
  __int128 v76; // [rsp-10h] [rbp-90h]
  __int128 v77; // [rsp-10h] [rbp-90h]
  unsigned __int64 v78; // [rsp+8h] [rbp-78h]
  unsigned __int64 v79; // [rsp+8h] [rbp-78h]
  unsigned __int64 v80; // [rsp+10h] [rbp-70h]
  unsigned __int64 v81; // [rsp+10h] [rbp-70h]
  __int64 v82; // [rsp+10h] [rbp-70h]
  __int64 v83; // [rsp+10h] [rbp-70h]
  __int64 v84; // [rsp+18h] [rbp-68h]
  __int64 v85; // [rsp+18h] [rbp-68h]
  __int64 v86; // [rsp+20h] [rbp-60h]
  __int64 v87; // [rsp+20h] [rbp-60h]
  const void **v88; // [rsp+20h] [rbp-60h]
  const void **v89; // [rsp+20h] [rbp-60h]
  __int64 v90; // [rsp+28h] [rbp-58h]
  __int64 v91; // [rsp+28h] [rbp-58h]
  __int64 v92; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v93; // [rsp+38h] [rbp-48h]
  __int64 v94; // [rsp+40h] [rbp-40h]
  unsigned int v95; // [rsp+48h] [rbp-38h]

  v9 = sub_1D18C40(a2, 1);
  if ( v9 )
  {
    if ( (unsigned __int8)sub_1D18C40(a2, 0) )
      return 0;
  }
  else
  {
    v11 = *(char **)(a2 + 40);
    v12 = *v11;
    if ( !*(_BYTE *)(a1 + 24)
      || ((v24 = *(_QWORD *)(a1 + 8), v25 = 1, v12 == 1) || v12 && (v25 = v12, *(_QWORD *)(v24 + 8LL * v12 + 120)))
      && (*(_BYTE *)(v24 + 259 * v25 + 2476) & 0xFB) == 0 )
    {
      v13 = (const void **)*((_QWORD *)v11 + 1);
      v14 = *(__int64 **)a1;
      v15 = *(_QWORD *)(a2 + 32);
      v16 = v12;
      v17 = *(unsigned int *)(a2 + 56);
      v18 = *(_QWORD *)(a2 + 72);
      v92 = v18;
      if ( v18 )
      {
        v80 = v12;
        v86 = v15;
        v90 = v17;
        sub_1623A60((__int64)&v92, v18, 2);
        v16 = v80;
        v15 = v86;
        v17 = v90;
      }
      *((_QWORD *)&v74 + 1) = v17;
      *(_QWORD *)&v74 = v15;
      v93 = *(_DWORD *)(a2 + 64);
      v19 = sub_1D44290(v14, 54, (__int64)&v92, v16, v13, a4, *(double *)a5.m128_u64, a6, v10, v74);
      goto LABEL_6;
    }
    if ( (unsigned __int8)sub_1D18C40(a2, 0) )
    {
      v27 = *(_QWORD *)(a2 + 72);
      v28 = *(__int64 **)a1;
      v29 = *(_QWORD *)(a2 + 32);
      v30 = *(unsigned int *)(a2 + 56);
      v31 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
      v32 = **(unsigned __int8 **)(a2 + 40);
      v92 = v27;
      v33 = v30;
      if ( v27 )
      {
        v81 = v32;
        v87 = v29;
        v91 = v30;
        sub_1623A60((__int64)&v92, v27, 2);
        v32 = v81;
        v29 = v87;
        v33 = v91;
      }
      *((_QWORD *)&v75 + 1) = v33;
      *(_QWORD *)&v75 = v29;
      v93 = *(_DWORD *)(a2 + 64);
      v34 = sub_1D44290(v28, 54, (__int64)&v92, v32, v31, a4, *(double *)a5.m128_u64, a6, v26, v75);
      if ( v92 )
        sub_161E7C0((__int64)&v92, v92);
      sub_1F81BC0(a1, (__int64)v34);
      *((_QWORD *)&v35 + 1) = v34;
      *(_QWORD *)&v35 = a1;
      v40 = sub_1FD0C90(v35, v36, v37, v38, v39, (__m128i)a4, a5, a6);
      if ( v40 && v34 != v40 )
      {
        if ( !*(_BYTE *)(a1 + 24)
          || ((v42 = *(_QWORD *)(a1 + 8), v43 = *(unsigned __int8 *)(v40[5] + 16LL * v41), (_BYTE)v43 == 1)
           || (_BYTE)v43 && *(_QWORD *)(v42 + 8LL * (unsigned __int8)v43 + 120))
          && (v44 = *((unsigned __int16 *)v40 + 12), (unsigned int)v44 <= 0x102)
          && !*(_BYTE *)(v44 + 259 * v43 + v42 + 2422) )
        {
          v92 = (__int64)v40;
          v93 = v41;
          v94 = (__int64)v40;
LABEL_52:
          v95 = v41;
          return sub_1F994A0(a1, a2, &v92, 2, 1);
        }
      }
      return 0;
    }
  }
  if ( !*(_BYTE *)(a1 + 24) )
  {
    v45 = *(_QWORD *)(a2 + 40);
    v46 = *(_BYTE *)(v45 + 16);
LABEL_30:
    v47 = *(const void ***)(v45 + 24);
    v48 = *(__int64 **)a1;
    v49 = *(_QWORD *)(a2 + 32);
    v50 = v46;
    v51 = *(unsigned int *)(a2 + 56);
    v52 = *(_QWORD *)(a2 + 72);
    v92 = v52;
    if ( v52 )
    {
      v78 = v46;
      v82 = v49;
      v84 = v51;
      v88 = v47;
      sub_1623A60((__int64)&v92, v52, 2);
      v50 = v78;
      v49 = v82;
      v51 = v84;
      v47 = v88;
    }
    *((_QWORD *)&v76 + 1) = v51;
    *(_QWORD *)&v76 = v49;
    v93 = *(_DWORD *)(a2 + 64);
    v19 = sub_1D44290(v48, a3, (__int64)&v92, v50, v47, a4, *(double *)a5.m128_u64, a6, v26, v76);
LABEL_6:
    v21 = (__int64)v19;
    v22 = v20;
    if ( v92 )
      sub_161E7C0((__int64)&v92, v92);
    v92 = v21;
    v93 = v22;
    v94 = v21;
    v95 = v22;
    return sub_1F994A0(a1, a2, &v92, 2, 1);
  }
  v45 = *(_QWORD *)(a2 + 40);
  v53 = *(_QWORD *)(a1 + 8);
  v54 = 1;
  v46 = *(_BYTE *)(v45 + 16);
  if ( v46 == 1 || v46 && (v54 = v46, *(_QWORD *)(v53 + 8LL * v46 + 120)) )
  {
    if ( !*(_BYTE *)(a3 + 259 * v54 + v53 + 2422) )
      goto LABEL_30;
  }
  if ( v9 )
  {
    v55 = *(const void ***)(v45 + 24);
    v56 = *(__int64 **)a1;
    v57 = *(_QWORD *)(a2 + 32);
    v58 = v46;
    v59 = *(unsigned int *)(a2 + 56);
    v60 = *(_QWORD *)(a2 + 72);
    v92 = v60;
    if ( v60 )
    {
      v79 = v46;
      v83 = v57;
      v85 = v59;
      v89 = v55;
      sub_1623A60((__int64)&v92, v60, 2);
      v58 = v79;
      v57 = v83;
      v59 = v85;
      v55 = v89;
    }
    *((_QWORD *)&v77 + 1) = v59;
    *(_QWORD *)&v77 = v57;
    v93 = *(_DWORD *)(a2 + 64);
    v61 = sub_1D44290(v56, a3, (__int64)&v92, v58, v55, a4, *(double *)a5.m128_u64, a6, v26, v77);
    v63 = v62;
    if ( v92 )
      sub_161E7C0((__int64)&v92, v92);
    sub_1F81BC0(a1, (__int64)v61);
    *((_QWORD *)&v64 + 1) = v61;
    *(_QWORD *)&v64 = a1;
    v69 = sub_1FD0C90(v64, v65, v66, v67, v68, (__m128i)a4, a5, a6);
    v70 = (__int64)v69;
    if ( v69 )
    {
      if ( v61 != v69 || v41 != v63 )
      {
        if ( !*(_BYTE *)(a1 + 24)
          || ((v71 = *(_QWORD *)(a1 + 8), v72 = *(unsigned __int8 *)(v69[5] + 16LL * v41), (_BYTE)v72 == 1)
           || (_BYTE)v72 && *(_QWORD *)(v71 + 8LL * (unsigned __int8)v72 + 120))
          && (v73 = *(unsigned __int16 *)(v70 + 24), (unsigned int)v73 <= 0x102)
          && !*(_BYTE *)(v73 + 259 * v72 + v71 + 2422) )
        {
          v92 = v70;
          v93 = v41;
          v94 = v70;
          goto LABEL_52;
        }
      }
    }
  }
  return 0;
}
