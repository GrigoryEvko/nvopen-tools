// Function: sub_1B63B00
// Address: 0x1b63b00
//
__int64 __fastcall sub_1B63B00(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 **a4,
        _QWORD *a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        __m128i a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        unsigned int *a15)
{
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v19; // rbx
  __int64 v22; // rdi
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  unsigned int v25; // edx
  __int64 v26; // r14
  __int64 v27; // r8
  unsigned int v28; // ecx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v35; // r8
  __int64 v36; // rsi
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // rdx
  int v40; // eax
  __int64 *v41; // rax
  __int64 v42; // rsi
  double v43; // xmm4_8
  double v44; // xmm5_8
  double v45; // xmm4_8
  double v46; // xmm5_8
  __int64 *v47; // rax
  unsigned __int64 v48; // rax
  _QWORD *v49; // r12
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // rbx
  __int64 *v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // rbx
  _QWORD *v61; // rax
  __int64 v62; // r9
  __int64 v63; // r13
  bool v64; // al
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  unsigned int *v68; // rsi
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rcx
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // [rsp+8h] [rbp-D8h]
  __int64 v78; // [rsp+10h] [rbp-D0h]
  __int64 v80; // [rsp+18h] [rbp-C8h]
  __int64 v82; // [rsp+20h] [rbp-C0h]
  __int64 v83; // [rsp+20h] [rbp-C0h]
  __int64 v84; // [rsp+20h] [rbp-C0h]
  __int64 v86; // [rsp+28h] [rbp-B8h]
  unsigned int *v87; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v88; // [rsp+38h] [rbp-A8h]
  _BYTE v89[32]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v90; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v91[14]; // [rsp+70h] [rbp-70h] BYREF

  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(_QWORD *)(v15 + 48);
  if ( !v16 )
    BUG();
  if ( *(_BYTE *)(v16 - 8) == 77 )
    goto LABEL_21;
  v17 = *(_QWORD *)(a1 + 8);
  if ( !v17 )
    goto LABEL_21;
  if ( *(_QWORD *)(v17 + 8) )
    goto LABEL_21;
  v19 = *(_QWORD *)(a1 - 48);
  v78 = *(_QWORD *)(a1 - 24);
  v22 = sub_157F0B0(v15);
  if ( !v22 )
    goto LABEL_21;
  v23 = sub_157EBA0(v22);
  if ( *(_BYTE *)(v23 + 16) != 27 )
    goto LABEL_21;
  v77 = v23;
  v24 = (_QWORD *)sub_13CF970(v23);
  if ( v19 != *v24 )
    goto LABEL_21;
  LOBYTE(v19) = v24[3] == 0 || v15 != v24[3];
  if ( !(_BYTE)v19 )
  {
    sub_1B46A00(v77, v78);
    if ( v39 != 4294967294LL )
    {
      v40 = *(unsigned __int16 *)(a1 + 18);
      BYTE1(v40) &= ~0x80u;
      if ( v40 == 32 )
      {
        v47 = (__int64 *)sub_157E9C0(v15);
        v42 = sub_159C540(v47);
      }
      else
      {
        v41 = (__int64 *)sub_157E9C0(v15);
        v42 = sub_159C4F0(v41);
      }
      LODWORD(v19) = 1;
      sub_164D160(
        a1,
        v42,
        a7,
        *(double *)a8.m128i_i64,
        *(double *)a9.m128i_i64,
        *(double *)a10.m128i_i64,
        v43,
        v44,
        a13,
        a14);
      sub_15F20C0((_QWORD *)a1);
      sub_1B5E140(v15, a4, a5, a6, a15, 0, a7, a8, a9, a10, v45, v46, a13, a14);
      return (unsigned int)v19;
    }
    v48 = sub_157EBA0(v15);
    v86 = sub_15F4DF0(v48, 0);
    v49 = sub_1648700(*(_QWORD *)(a1 + 8));
    if ( *((_BYTE *)v49 + 16) == 77 )
    {
      v50 = *(_QWORD *)(v86 + 48);
      if ( v50 )
      {
        if ( v49 == (_QWORD *)(v50 - 24) )
        {
          v51 = v49[4];
          if ( !v51 )
            BUG();
          if ( *(_BYTE *)(v51 - 8) != 77 )
          {
            v52 = (__int64 *)sub_157E9C0(v15);
            v53 = sub_159C4F0(v52);
            v54 = (__int64 *)sub_157E9C0(v15);
            v80 = sub_159C540(v54);
            v57 = *(unsigned __int16 *)(a1 + 18);
            BYTE1(v57) &= ~0x80u;
            if ( v57 == 32 )
            {
              v58 = v53;
              v53 = v80;
              v80 = v58;
            }
            sub_164D160(
              a1,
              v53,
              a7,
              *(double *)a8.m128i_i64,
              *(double *)a9.m128i_i64,
              *(double *)a10.m128i_i64,
              v55,
              v56,
              a13,
              a14);
            sub_15F20C0((_QWORD *)a1);
            v59 = *(_QWORD *)(v15 + 56);
            LOWORD(v91[0]) = 259;
            v82 = v59;
            v90.m128i_i64[0] = (__int64)"switch.edge";
            v60 = sub_157E9C0(v15);
            v61 = (_QWORD *)sub_22077B0(64);
            v62 = v77;
            v63 = (__int64)v61;
            if ( v61 )
            {
              sub_157FB60(v61, v60, (__int64)&v90, v82, v15);
              v62 = v77;
            }
            v90.m128i_i64[0] = (__int64)v91;
            v83 = v62;
            v90.m128i_i64[1] = 0x800000000LL;
            v64 = sub_1B43680(v62);
            v67 = v83;
            if ( v64 )
            {
              sub_1B43970(v83, (__int64)&v90);
              v67 = v83;
              if ( (*(_DWORD *)(v83 + 20) & 0xFFFFFFFu) >> 1 == v90.m128i_i32[2] )
              {
                *(_QWORD *)v90.m128i_i64[0] = (unsigned __int64)(*(_QWORD *)v90.m128i_i64[0] + 1LL) >> 1;
                sub_1525CA0((__int64)&v90, v90.m128i_i64[0]);
                v87 = (unsigned int *)v89;
                v88 = 0x800000000LL;
                sub_1B4B9A0((__int64)&v87, v90.m128i_i64[0], v90.m128i_i64[0] + 8LL * v90.m128i_u32[2], v74, v75, v76);
                sub_1B42940(v83, v87, (unsigned int)v88);
                v67 = v83;
                if ( v87 != (unsigned int *)v89 )
                {
                  _libc_free((unsigned __int64)v87);
                  v67 = v83;
                }
              }
            }
            v84 = v67;
            sub_15FFFB0(v67, v78, v63, v65, v66, v67);
            a2[1] = v63;
            a2[2] = v63 + 40;
            v68 = *(unsigned int **)(v84 + 48);
            v87 = v68;
            if ( v68 )
            {
              sub_1623A60((__int64)&v87, (__int64)v68, 2);
              v69 = *a2;
              if ( !*a2 )
                goto LABEL_45;
            }
            else
            {
              v69 = *a2;
              if ( !*a2 )
              {
LABEL_47:
                sub_17CD270((__int64 *)&v87);
                sub_1B44660(a2, v86);
                sub_1704F80((__int64)v49, v80, v63, v71, v72, v73);
                if ( (_QWORD *)v90.m128i_i64[0] != v91 )
                  _libc_free(v90.m128i_u64[0]);
                LODWORD(v19) = 1;
                return (unsigned int)v19;
              }
            }
            sub_161E7C0((__int64)a2, v69);
LABEL_45:
            v70 = (unsigned __int8 *)v87;
            *a2 = (__int64)v87;
            if ( v70 )
            {
              sub_1623210((__int64)&v87, v70, (__int64)a2);
              v87 = 0;
            }
            goto LABEL_47;
          }
        }
      }
    }
LABEL_21:
    LODWORD(v19) = 0;
    return (unsigned int)v19;
  }
  v25 = (*(_DWORD *)(v77 + 20) & 0xFFFFFFFu) >> 1;
  v26 = v25 - 1;
  if ( v25 == 1 )
  {
LABEL_23:
    v27 = 0;
  }
  else
  {
    v27 = 0;
    v28 = 2;
    v29 = 1;
    do
    {
      v31 = 24;
      if ( (_DWORD)v29 != -1 )
        v31 = 24LL * (v28 + 1);
      v32 = v77 - 24LL * (*(_DWORD *)(v77 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v77 + 23) & 0x40) != 0 )
        v32 = *(_QWORD *)(v77 - 8);
      v33 = *(_QWORD *)(v32 + v31);
      if ( v15 == v33 && v33 )
      {
        if ( v27 )
          goto LABEL_23;
        v30 = v29;
        v27 = *(_QWORD *)(v32 + 24LL * v28);
      }
      else
      {
        v30 = v29;
      }
      ++v29;
      v28 += 2;
    }
    while ( v30 != v26 );
  }
  sub_1593B40((_QWORD *)(a1 - 48), v27);
  v91[0] = 0;
  v90 = (__m128i)a3;
  v91[1] = 0;
  v91[2] = a1;
  v36 = sub_13E3350(a1, &v90, 0, 1, v35);
  if ( v36 )
  {
    sub_164D160(
      a1,
      v36,
      a7,
      *(double *)a8.m128i_i64,
      *(double *)a9.m128i_i64,
      *(double *)a10.m128i_i64,
      v37,
      v38,
      a13,
      a14);
    sub_15F20C0((_QWORD *)a1);
  }
  sub_1B5E140(v15, a4, a5, a6, a15, 0, a7, a8, a9, a10, v37, v38, a13, a14);
  return (unsigned int)v19;
}
