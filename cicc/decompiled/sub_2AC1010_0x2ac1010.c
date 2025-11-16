// Function: sub_2AC1010
// Address: 0x2ac1010
//
__int64 __fastcall sub_2AC1010(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int64 a4)
{
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // r14
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // edx
  unsigned __int8 *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // r8
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rbx
  __int64 v22; // r9
  _BYTE *v23; // rdi
  int v24; // ecx
  _BYTE *v25; // rsi
  __int64 v26; // rcx
  unsigned int v27; // edx
  _QWORD *v28; // rcx
  __int64 v29; // rdi
  int v30; // esi
  __int64 v31; // rax
  __int64 v32; // rcx
  _BOOL4 v33; // edx
  _BOOL4 v34; // esi
  unsigned __int64 v35; // rax
  _BYTE *v36; // rdi
  __int64 v38; // rdx
  _BYTE *v39; // rbx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  bool v46; // of
  unsigned __int64 v47; // rbx
  signed __int64 v48; // rax
  int v49; // edx
  bool v50; // zf
  int v51; // edx
  unsigned __int64 v52; // rbx
  int v53; // eax
  bool v54; // cc
  unsigned __int64 v55; // rax
  __int64 v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+10h] [rbp-C0h]
  int v58; // [rsp+18h] [rbp-B8h]
  int v59; // [rsp+28h] [rbp-A8h]
  __int64 v60; // [rsp+30h] [rbp-A0h]
  __int64 v61; // [rsp+30h] [rbp-A0h]
  __int64 v62; // [rsp+38h] [rbp-98h]
  unsigned int v63; // [rsp+44h] [rbp-8Ch]
  signed __int64 v64; // [rsp+48h] [rbp-88h]
  int v65; // [rsp+48h] [rbp-88h]
  int v66; // [rsp+50h] [rbp-80h]
  _BYTE *v67; // [rsp+50h] [rbp-80h]
  _BOOL4 v68; // [rsp+50h] [rbp-80h]
  __int64 v70; // [rsp+60h] [rbp-70h] BYREF
  __int64 v71; // [rsp+68h] [rbp-68h]
  __int64 v72; // [rsp+70h] [rbp-60h] BYREF
  __int64 v73; // [rsp+78h] [rbp-58h]
  _BYTE v74[80]; // [rsp+80h] [rbp-50h] BYREF

  if ( BYTE4(a4) )
  {
    v66 = 1;
    v64 = 0;
  }
  else
  {
    v70 = sub_DFD270(*(_QWORD *)(a2 + 448), 55, *(_DWORD *)(a2 + 992));
    v72 = _mm_cvtsi32_si128(a4).m128i_u64[0];
    v67 = (_BYTE *)v72;
    v71 = v38;
    LODWORD(v73) = 0;
    sub_2AA9150((__int64)&v72, (__int64)&v70);
    v39 = (_BYTE *)v72;
    v65 = v73;
    v70 = sub_DFD800(
            *(_QWORD *)(a2 + 448),
            (unsigned int)*a3 - 29,
            *((_QWORD *)a3 + 1),
            *(_DWORD *)(a2 + 992),
            v40,
            v41,
            0,
            0,
            0,
            0);
    v71 = v42;
    LODWORD(v73) = 0;
    v72 = (__int64)v67;
    sub_2AA9150((__int64)&v72, (__int64)&v70);
    v68 = v73;
    if ( (_DWORD)v73 != 1 )
      v68 = v65 == 1;
    v46 = __OFADD__(v72, v39);
    v47 = (unsigned __int64)&v39[v72];
    if ( v46 )
    {
      v47 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v72 <= 0 )
        v47 = 0x8000000000000000LL;
    }
    v48 = sub_2AC04F0(a2, (__int64)a3, a4, v43, v44, v45);
    v50 = v49 == 1;
    v51 = 1;
    if ( !v50 )
      v51 = v68;
    v46 = __OFADD__(v48, v47);
    v52 = v48 + v47;
    v66 = v51;
    if ( v46 )
    {
      v54 = v48 <= 0;
      v55 = 0x8000000000000000LL;
      if ( !v54 )
        v55 = 0x7FFFFFFFFFFFFFFFLL;
      v64 = v55;
    }
    else
    {
      v64 = v52;
    }
    if ( *(_DWORD *)(a2 + 992) != 2 )
      v64 /= 2LL;
  }
  v7 = sub_2AAEDF0(*((_QWORD *)a3 + 1), a4);
  v8 = *(__int64 **)(a2 + 448);
  v9 = v7;
  v10 = (_QWORD *)sub_BD5C60((__int64)a3);
  v11 = sub_BCB2A0(v10);
  sub_2AAEDF0(v11, a4);
  v12 = sub_DFD2D0(v8, 57, v9);
  v59 = v13;
  v62 = v12;
  v14 = *(unsigned __int8 **)(sub_986520((__int64)a3) + 32);
  v15 = sub_DFB770(v14);
  v63 = v15;
  v16 = v15;
  if ( !(_DWORD)v15 )
  {
    v61 = v15;
    v53 = (unsigned __int8)sub_31A5290(*(_QWORD *)(a2 + 440), v14);
    v16 = v61;
    v63 = (unsigned __int8)v53;
  }
  v60 = v16;
  v17 = sub_986520((__int64)a3);
  v18 = *((_DWORD *)a3 + 1);
  v73 = 0x400000000LL;
  v72 = (__int64)v74;
  v19 = v60;
  v20 = v18 & 0x7FFFFFF;
  v21 = v20;
  v22 = 32 * v20;
  if ( v20 > 4 )
  {
    v56 = 32 * v20;
    v57 = v17;
    v58 = v20;
    sub_C8D5F0((__int64)&v72, v74, v20, 8u, v60, v22);
    v25 = (_BYTE *)v72;
    v24 = v73;
    LODWORD(v20) = v58;
    v17 = v57;
    v22 = v56;
    v23 = (_BYTE *)(v72 + 8LL * (unsigned int)v73);
    v19 = v60;
  }
  else
  {
    v23 = v74;
    v24 = 0;
    v25 = v74;
  }
  if ( v22 )
  {
    v26 = 0;
    do
    {
      *(_QWORD *)&v23[v26] = *(_QWORD *)(v17 + 4 * v26);
      v26 += 8;
      --v21;
    }
    while ( v21 );
    v25 = (_BYTE *)v72;
    v24 = v73;
  }
  v27 = v24 + v20;
  v28 = v25;
  v29 = *(_QWORD *)(a2 + 448);
  v30 = *a3;
  LODWORD(v73) = v27;
  v31 = sub_DFD800(
          v29,
          v30 - 29,
          v9,
          *(_DWORD *)(a2 + 992),
          0,
          v63 | v19 & 0xFFFFFFFF00000000LL,
          v28,
          v27,
          (__int64)a3,
          0);
  v32 = v31;
  v34 = v33;
  if ( !v33 )
    v34 = v59 == 1;
  v35 = v31 + v62;
  if ( __OFADD__(v32, v62) )
  {
    v35 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v32 <= 0 )
      v35 = 0x8000000000000000LL;
  }
  v36 = (_BYTE *)v72;
  *(_QWORD *)(a1 + 16) = v35;
  *(_DWORD *)(a1 + 24) = v34;
  *(_QWORD *)a1 = v64;
  *(_DWORD *)(a1 + 8) = v66;
  if ( v36 != v74 )
    _libc_free((unsigned __int64)v36);
  return a1;
}
