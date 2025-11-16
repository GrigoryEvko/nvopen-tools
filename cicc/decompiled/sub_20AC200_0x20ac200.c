// Function: sub_20AC200
// Address: 0x20ac200
//
__int64 *__fastcall sub_20AC200(
        __int64 *a1,
        __int64 a2,
        const void **a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int64 a11,
        __int64 a12)
{
  unsigned int v12; // r10d
  unsigned __int64 v14; // r13
  __int64 v16; // r15
  bool v17; // zf
  __int64 v18; // r11
  __int16 v19; // si
  int v20; // edx
  __int64 v21; // r12
  __int64 v22; // rcx
  char v23; // al
  const void **v24; // rcx
  __int64 v26; // rcx
  const void **v27; // rcx
  bool v28; // al
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 *v32; // rdi
  __m128i v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  signed int v36; // esi
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 (*v43)(); // rax
  int v44; // eax
  __int64 v45; // rsi
  unsigned __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 *v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // r8
  __int16 *v51; // r9
  __int128 v52; // xmm0
  __int64 v53; // rax
  unsigned int v54; // ebx
  unsigned __int64 v55; // [rsp+8h] [rbp-C8h]
  __int64 v56; // [rsp+18h] [rbp-B8h]
  __m128i v57; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+30h] [rbp-A0h]
  __int64 v59; // [rsp+38h] [rbp-98h]
  __int128 v60; // [rsp+40h] [rbp-90h]
  __int64 *v61; // [rsp+50h] [rbp-80h]
  __int64 v62; // [rsp+58h] [rbp-78h]
  __int64 *v63; // [rsp+60h] [rbp-70h]
  __int16 *v64; // [rsp+68h] [rbp-68h]
  const void **v65; // [rsp+70h] [rbp-60h]
  __int64 v66; // [rsp+78h] [rbp-58h]
  unsigned int v67; // [rsp+80h] [rbp-50h] BYREF
  const void **v68; // [rsp+88h] [rbp-48h]
  __int64 v69; // [rsp+90h] [rbp-40h] BYREF
  int v70; // [rsp+98h] [rbp-38h]

  v12 = a5;
  v14 = a4;
  v16 = a10;
  v63 = a1;
  v65 = a3;
  v17 = *(_WORD *)(a10 + 24) == 118;
  v18 = a11;
  v62 = a2;
  v66 = a12;
  v19 = *(_WORD *)(a4 + 24);
  v20 = DWORD2(a10);
  if ( v17 )
  {
    if ( v19 != 118 )
    {
      v12 = DWORD2(a10);
      v14 = a10;
      v16 = a4;
      v20 = a5;
    }
    v21 = 16LL * v12;
    v26 = v21 + *(_QWORD *)(v14 + 40);
    v23 = *(_BYTE *)v26;
    v27 = *(const void ***)(v26 + 8);
    LOBYTE(v67) = v23;
    v68 = v27;
  }
  else
  {
    v21 = 16LL * (unsigned int)a5;
    v22 = v21 + *(_QWORD *)(a4 + 40);
    v23 = *(_BYTE *)v22;
    v24 = *(const void ***)(v22 + 8);
    LOBYTE(v67) = v23;
    v68 = v24;
    if ( v19 != 118 )
      return 0;
  }
  if ( v23 )
  {
    v28 = (unsigned __int8)(v23 - 2) <= 5u || (unsigned __int8)(v23 - 14) <= 0x47u;
  }
  else
  {
    v59 = a11;
    LODWORD(v61) = v12;
    LODWORD(v60) = v20;
    v28 = sub_1F58CF0((__int64)&v67);
    v18 = a11;
    v12 = (unsigned int)v61;
    v20 = v60;
  }
  if ( !v28 || a6 != 17 && a6 != 22 )
    return 0;
  v60 = 0u;
  v29 = *(_QWORD *)(v14 + 32);
  if ( *(_QWORD *)v29 == v16 && *(_DWORD *)(v29 + 8) == v20 )
  {
    v41 = *(_QWORD *)(v29 + 40);
    v42 = *(unsigned int *)(v29 + 8);
    *(_QWORD *)&v60 = v16;
    v56 = v41;
    v55 = *(unsigned int *)(v29 + 48);
    *((_QWORD *)&v60 + 1) = v42 | *((_QWORD *)&v60 + 1) & 0xFFFFFFFF00000000LL;
  }
  else
  {
    if ( *(_QWORD *)(v29 + 40) != v16 || *(_DWORD *)(v29 + 48) != v20 )
      return 0;
    v30 = *(_QWORD *)v29;
    v31 = *(unsigned int *)(v29 + 48);
    *(_QWORD *)&v60 = v16;
    v56 = v30;
    v55 = *(unsigned int *)(v29 + 8);
    *((_QWORD *)&v60 + 1) = v31 | *((_QWORD *)&v60 + 1) & 0xFFFFFFFF00000000LL;
  }
  v32 = *(__int64 **)(v18 + 16);
  LODWORD(v59) = v12;
  v58 = v18;
  v61 = v32;
  v33.m128i_i64[0] = sub_1D38BB0((__int64)v32, 0, v66, v67, v68, 0, a7, a8, a9, 0);
  *(_QWORD *)&v60 = v16;
  v57 = v33;
  if ( (unsigned __int8)sub_1D208B0((__int64)v32, v16, *((__int64 *)&v60 + 1)) )
  {
    LODWORD(v60) = v59;
    v36 = sub_1D16EF0(a6, 1);
    if ( *(int *)(v58 + 8) > 1 )
    {
      v34 = ((*(_BYTE *)(*(_QWORD *)(v14 + 40) + v21) >> 3) & 0x1F) + 15LL * v36 + 18112;
      v35 = 4 * (*(_BYTE *)(*(_QWORD *)(v14 + 40) + v21) & 7u);
      if ( ((*((_DWORD *)v63 + v34 + 3) >> (4 * (*(_BYTE *)(*(_QWORD *)(v14 + 40) + v21) & 7))) & 0xF) != 0 )
        return 0;
    }
    v39 = sub_1D28D50(v61, v36, v34, v35, v37, v38);
    return sub_1D3A900(
             v61,
             0x89u,
             v66,
             (unsigned int)v62,
             v65,
             0,
             (__m128)a7,
             a8,
             a9,
             v14,
             (__int16 *)((unsigned int)v60 | a5 & 0xFFFFFFFF00000000LL),
             *(_OWORD *)&v57,
             v39,
             v40);
  }
  else
  {
    if ( !sub_1D18C00(v14, 1, v59) )
      return 0;
    v43 = *(__int64 (**)())(*v63 + 216);
    if ( v43 == sub_1F3CA20
      || !((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD))v43)(v63, v16, *((_QWORD *)&v60 + 1)) )
    {
      return 0;
    }
    v44 = *(unsigned __int16 *)(v16 + 24);
    if ( v44 == 32 || v44 == 10 )
    {
      v53 = *(_QWORD *)(v16 + 88);
      v54 = *(_DWORD *)(v53 + 32);
      if ( v54 <= 0x40 )
      {
        if ( !*(_QWORD *)(v53 + 24) )
          return 0;
      }
      else if ( v54 == (unsigned int)sub_16A57B0(v53 + 24) )
      {
        return 0;
      }
    }
    v45 = *(_QWORD *)(v56 + 72);
    v69 = v45;
    if ( v45 )
      sub_1623A60((__int64)&v69, v45, 2);
    v70 = *(_DWORD *)(v56 + 64);
    v63 = sub_1D3C080(v61, (__int64)&v69, v56, v55, v67, v68, a7, a8, a9);
    v64 = (__int16 *)v46;
    if ( v69 )
      sub_161E7C0((__int64)&v69, v69);
    v47 = *(_QWORD *)(v14 + 72);
    v69 = v47;
    if ( v47 )
      sub_1623A60((__int64)&v69, v47, 2);
    v70 = *(_DWORD *)(v14 + 64);
    v48 = sub_1D332F0(
            v61,
            118,
            (__int64)&v69,
            v67,
            v68,
            0,
            *(double *)a7.m128i_i64,
            a8,
            a9,
            (__int64)v63,
            (unsigned __int64)v64,
            v60);
    v50 = (unsigned __int64)v48;
    v51 = (__int16 *)v49;
    if ( v69 )
    {
      v63 = v48;
      v64 = (__int16 *)v49;
      sub_161E7C0((__int64)&v69, v69);
      v50 = (unsigned __int64)v63;
      v51 = v64;
    }
    v52 = (__int128)_mm_load_si128(&v57);
    return sub_1F81070(v61, v66, (unsigned int)v62, v65, v50, v51, (__m128)v52, a8, a9, v52, a6);
  }
}
