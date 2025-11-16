// Function: sub_32AD6C0
// Address: 0x32ad6c0
//
__int64 __fastcall sub_32AD6C0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  int v9; // r13d
  unsigned __int16 *v10; // rax
  unsigned __int16 v11; // r8
  unsigned int v12; // r12d
  unsigned int v13; // r15d
  __int64 v14; // rcx
  __int64 v15; // rax
  bool v17; // al
  __int128 v18; // kr00_16
  unsigned __int16 v19; // r8
  bool v20; // al
  int v21; // r9d
  __int64 v22; // rdx
  unsigned __int16 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  bool v28; // al
  __int128 v29; // kr20_16
  unsigned __int16 v30; // r8
  bool v31; // al
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int16 v34; // [rsp+0h] [rbp-100h]
  unsigned __int16 v35; // [rsp+0h] [rbp-100h]
  unsigned __int16 v36; // [rsp+18h] [rbp-E8h]
  unsigned __int16 v37; // [rsp+18h] [rbp-E8h]
  __int64 v38; // [rsp+28h] [rbp-D8h]
  char v40; // [rsp+4Bh] [rbp-B5h]
  int v41; // [rsp+4Ch] [rbp-B4h]
  __m128i v42; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v43; // [rsp+60h] [rbp-A0h] BYREF
  int v44; // [rsp+70h] [rbp-90h] BYREF
  __int64 v45; // [rsp+78h] [rbp-88h]
  int v46; // [rsp+80h] [rbp-80h]
  __int64 v47; // [rsp+88h] [rbp-78h]
  int v48; // [rsp+90h] [rbp-70h]
  char v49; // [rsp+9Ch] [rbp-64h]
  int v50; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-58h]
  int v52; // [rsp+B0h] [rbp-50h]
  __int64 v53; // [rsp+B8h] [rbp-48h]
  int v54; // [rsp+C0h] [rbp-40h]
  char v55; // [rsp+CCh] [rbp-34h]

  v42.m128i_i64[0] = a4;
  v43.m128i_i64[1] = a3;
  v42.m128i_i64[1] = a5;
  v43.m128i_i64[0] = a2;
  v41 = a5;
  v9 = a3;
  v10 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v12 = (a6 - 18 > 3) + 178;
  v38 = *((_QWORD *)v10 + 1);
  v13 = *v10;
  v40 = *((_BYTE *)a1 + 33);
  if ( v40 )
  {
    v14 = a1[1];
    v15 = 1;
    if ( v11 != 1 )
    {
      if ( !v11 )
        return 0;
      v15 = v11;
      if ( !*(_QWORD *)(v14 + 8LL * v11 + 112) )
        return 0;
    }
    if ( *(_BYTE *)(v12 + v14 + 500 * v15 + 6414) )
      return 0;
  }
  if ( a6 > 0x13 )
  {
    if ( a6 - 20 > 1 )
      return 0;
  }
  else
  {
    if ( a6 > 0x11 )
      goto LABEL_20;
    if ( a6 <= 0xB )
    {
      if ( a6 <= 9 )
        return 0;
LABEL_20:
      v35 = v11;
      v47 = a4;
      v44 = 57;
      v45 = a2;
      v46 = a3;
      v48 = v41;
      v49 = 0;
      v28 = sub_32AD640(a7, *((__int64 *)&a7 + 1), 0, (__int64)&v44);
      v29 = a7;
      v30 = v35;
      if ( !v28 )
        goto LABEL_22;
      v51 = a4;
      v50 = 57;
      v52 = v41;
      v53 = a2;
      v54 = v9;
      v55 = 0;
      v31 = sub_32AD640(a8, *((__int64 *)&a8 + 1), 0, (__int64)&v50);
      v30 = v35;
      v29 = a7;
      if ( !v31 )
      {
LABEL_22:
        v37 = v30;
        v45 = a4;
        v44 = 57;
        v46 = v41;
        v47 = a2;
        v48 = v9;
        v49 = 0;
        if ( !sub_32AD640(v29, *((__int64 *)&v29 + 1), 0, (__int64)&v44) )
          return 0;
        v53 = a4;
        v50 = 57;
        v51 = a2;
        v52 = v9;
        v54 = v41;
        v55 = 0;
        if ( !sub_32AD640(a8, *((__int64 *)&a8 + 1), 0, (__int64)&v50) )
          return 0;
        v22 = a1[1];
        v23 = v37;
        if ( !v40 )
        {
          v32 = 1;
          if ( v37 != 1 )
          {
            if ( !v37 )
              return 0;
            v32 = v37;
            if ( !*(_QWORD *)(v22 + 8LL * v37 + 112) )
              return 0;
          }
          if ( (*(_BYTE *)(v12 + 500 * v32 + v22 + 6414) & 0xFB) != 0 )
            return 0;
          goto LABEL_17;
        }
        goto LABEL_33;
      }
      return sub_3406EB0(
               *a1,
               v12,
               a9,
               v13,
               v38,
               (unsigned int)&v44,
               *(_OWORD *)&_mm_load_si128(&v43),
               *(_OWORD *)&_mm_load_si128(&v42));
    }
    if ( a6 - 12 > 1 )
      return 0;
  }
  v34 = v11;
  v45 = a4;
  v44 = 57;
  v46 = v41;
  v47 = a2;
  v48 = a3;
  v49 = 0;
  v17 = sub_32AD640(a7, *((__int64 *)&a7 + 1), 0, (__int64)&v44);
  v18 = a7;
  v19 = v34;
  if ( !v17 )
    goto LABEL_12;
  v53 = a4;
  v50 = 57;
  v51 = a2;
  v52 = v9;
  v54 = v41;
  v55 = 0;
  v20 = sub_32AD640(a8, *((__int64 *)&a8 + 1), 0, (__int64)&v50);
  v19 = v34;
  v18 = a7;
  if ( !v20 )
  {
LABEL_12:
    v36 = v19;
    v47 = a4;
    v44 = 57;
    v45 = a2;
    v46 = v9;
    v48 = v41;
    v49 = 0;
    if ( !sub_32AD640(v18, *((__int64 *)&v18 + 1), 0, (__int64)&v44) )
      return 0;
    v51 = a4;
    v50 = 57;
    v52 = v41;
    v53 = a2;
    v54 = v9;
    v55 = 0;
    if ( !sub_32AD640(a8, *((__int64 *)&a8 + 1), 0, (__int64)&v50) )
      return 0;
    v22 = a1[1];
    v23 = v36;
    if ( !v40 )
    {
      v24 = 1;
      if ( v36 != 1 )
      {
        if ( !v36 )
          return 0;
        v24 = v36;
        if ( !*(_QWORD *)(v22 + 8LL * v36 + 112) )
          return 0;
      }
      if ( (*(_BYTE *)(v12 + v22 + 500 * v24 + 6414) & 0xFB) != 0 )
        return 0;
LABEL_17:
      v25 = *a1;
      v26 = sub_3406EB0(*a1, v12, a9, v13, v38, v21, *(_OWORD *)&v43, *(_OWORD *)&v42);
      return sub_3407430(v25, v26, v27, a9, v13, v38);
    }
LABEL_33:
    v33 = 1;
    if ( v23 != 1 )
    {
      if ( !v23 )
        return 0;
      v33 = v23;
      if ( !*(_QWORD *)(v22 + 8LL * v23 + 112) )
        return 0;
    }
    if ( *(_BYTE *)(v12 + 500 * v33 + v22 + 6414) )
      return 0;
    goto LABEL_17;
  }
  return sub_3406EB0(
           *a1,
           v12,
           a9,
           v13,
           v38,
           (unsigned int)&v44,
           *(_OWORD *)&_mm_load_si128(&v43),
           *(_OWORD *)&_mm_load_si128(&v42));
}
