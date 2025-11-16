// Function: sub_1F8F600
// Address: 0x1f8f600
//
__int64 *__fastcall sub_1F8F600(__int64 **a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 **v5; // r11
  __int64 *v7; // rax
  unsigned __int8 *v8; // rcx
  __int64 v9; // r10
  char v10; // si
  __int64 v11; // r8
  int v12; // edx
  unsigned __int64 v13; // r9
  __int64 v14; // r14
  __int64 v15; // r15
  unsigned int v16; // ebx
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 v20; // rsi
  __int64 *v21; // r10
  __int64 *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rcx
  _QWORD *v27; // rdx
  __int64 *v28; // r15
  bool v29; // cl
  __int64 v30; // rsi
  __int128 v31; // rax
  unsigned int *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 *v36; // rax
  __int64 v37; // r10
  __int64 v38; // r11
  __int64 v39; // r14
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r15
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 *v44; // r11
  __int128 v45; // [rsp-10h] [rbp-80h]
  __int128 v46; // [rsp-10h] [rbp-80h]
  __int64 v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+8h] [rbp-68h]
  __int64 v49; // [rsp+8h] [rbp-68h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  bool v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+10h] [rbp-60h]
  __int64 v53; // [rsp+10h] [rbp-60h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  unsigned __int64 v56; // [rsp+18h] [rbp-58h]
  __int64 *v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+20h] [rbp-50h]
  __int64 *v59; // [rsp+20h] [rbp-50h]
  __int64 **v60; // [rsp+20h] [rbp-50h]
  __int64 *v61; // [rsp+20h] [rbp-50h]
  const void **v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+30h] [rbp-40h] BYREF
  int v64; // [rsp+38h] [rbp-38h]

  v5 = a1;
  v7 = *(__int64 **)(a2 + 32);
  v8 = *(unsigned __int8 **)(a2 + 40);
  v9 = *v7;
  v10 = *v8;
  v11 = *v7;
  v12 = *(unsigned __int16 *)(*v7 + 24);
  v13 = v7[1];
  v62 = (const void **)*((_QWORD *)v8 + 1);
  v14 = v7[5];
  v15 = v7[6];
  v16 = *v8;
  switch ( v12 )
  {
    case 11:
    case 33:
      v20 = *(_QWORD *)(a2 + 72);
      v21 = *a1;
      v63 = v20;
      if ( v20 )
      {
        v50 = v11;
        v56 = v13;
        v57 = v21;
        sub_1623A60((__int64)&v63, v20, 2);
        v11 = v50;
        v13 = v56;
        v21 = v57;
      }
      *((_QWORD *)&v45 + 1) = v15;
      *(_QWORD *)&v45 = v14;
      v64 = *(_DWORD *)(a2 + 64);
      v22 = sub_1D332F0(v21, 154, (__int64)&v63, v16, v62, 0, *(double *)a3.m128i_i64, a4, a5, v11, v13, v45);
      goto LABEL_12;
    case 157:
      v32 = *(unsigned int **)(v9 + 32);
      v33 = *(_QWORD *)(*(_QWORD *)v32 + 40LL) + 16LL * v32[2];
      if ( *(_BYTE *)v33 == v10 && (*(const void ***)(v33 + 8) == v62 || v10) )
        return *(__int64 **)v32;
      return sub_1F77270(a1, a2, *(double *)a3.m128i_i64, a4, a5);
    case 154:
      v23 = *(_QWORD *)(v7[5] + 88);
      v24 = *(_QWORD **)(v23 + 24);
      if ( *(_DWORD *)(v23 + 32) > 0x40u )
        v24 = (_QWORD *)*v24;
      v25 = *(_QWORD *)(v9 + 32);
      v26 = *(_QWORD *)(*(_QWORD *)(v25 + 40) + 88LL);
      v27 = *(_QWORD **)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v27 = (_QWORD *)*v27;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v25 + 40LL) + 16LL * *(unsigned int *)(v25 + 8)) == 11 && v10 == 8 )
        return 0;
      v28 = *a1;
      v29 = v27 == (_QWORD *)1;
      if ( (*(_BYTE *)(**a1 + 792) & 2) != 0 || v27 == (_QWORD *)1 )
      {
        v30 = *(_QWORD *)(a2 + 72);
        v63 = v30;
        if ( v30 )
        {
          v47 = v9;
          v51 = v27 == (_QWORD *)1;
          sub_1623A60((__int64)&v63, v30, 2);
          v9 = v47;
          v29 = v51;
          v28 = *a1;
        }
        v58 = v9;
        v64 = *(_DWORD *)(a2 + 64);
        *(_QWORD *)&v31 = sub_1D38E70(
                            (__int64)v28,
                            v29 & (unsigned __int8)(v24 == (_QWORD *)1),
                            (__int64)&v63,
                            0,
                            a3,
                            a4,
                            a5);
        v22 = sub_1D332F0(
                v28,
                154,
                (__int64)&v63,
                v16,
                v62,
                0,
                *(double *)a3.m128i_i64,
                a4,
                a5,
                **(_QWORD **)(v58 + 32),
                *(_QWORD *)(*(_QWORD *)(v58 + 32) + 8LL),
                v31);
        goto LABEL_12;
      }
      return sub_1F77270(a1, a2, *(double *)a3.m128i_i64, a4, a5);
  }
  if ( v12 != 101 )
    return sub_1F77270(a1, a2, *(double *)a3.m128i_i64, a4, a5);
  v17 = *(_QWORD *)(v9 + 48);
  if ( !v17 || *(_QWORD *)(v17 + 32) )
    return sub_1F77270(a1, a2, *(double *)a3.m128i_i64, a4, a5);
  v34 = *(_QWORD *)(v9 + 72);
  v35 = *(_QWORD *)(v9 + 32);
  v59 = *a1;
  v63 = v34;
  if ( v34 )
  {
    v48 = v9;
    v52 = v35;
    sub_1623A60((__int64)&v63, v34, 2);
    v5 = a1;
    v9 = v48;
    v35 = v52;
  }
  *((_QWORD *)&v46 + 1) = v15;
  *(_QWORD *)&v46 = v14;
  v64 = *(_DWORD *)(v9 + 64);
  v49 = (__int64)v5;
  v53 = v9;
  v36 = sub_1D332F0(
          v59,
          154,
          (__int64)&v63,
          v16,
          v62,
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          *(_QWORD *)v35,
          *(_QWORD *)(v35 + 8),
          v46);
  v37 = v53;
  v38 = v49;
  v39 = (__int64)v36;
  v41 = v40;
  if ( v63 )
  {
    sub_161E7C0((__int64)&v63, v63);
    v38 = v49;
    v37 = v53;
  }
  v54 = v37;
  v60 = (__int64 **)v38;
  sub_1F81BC0(v38, v39);
  v42 = *(_QWORD *)(a2 + 72);
  v43 = *(_QWORD *)(v54 + 32);
  v44 = *v60;
  v63 = v42;
  if ( v42 )
  {
    v55 = v43;
    v61 = v44;
    sub_1623A60((__int64)&v63, v42, 2);
    v43 = v55;
    v44 = v61;
  }
  v64 = *(_DWORD *)(a2 + 64);
  v22 = sub_1D332F0(
          v44,
          101,
          (__int64)&v63,
          v16,
          v62,
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v39,
          v41,
          *(_OWORD *)(v43 + 40));
LABEL_12:
  v18 = v22;
  if ( v63 )
    sub_161E7C0((__int64)&v63, v63);
  return v18;
}
