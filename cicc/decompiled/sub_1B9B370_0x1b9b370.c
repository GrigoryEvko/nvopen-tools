// Function: sub_1B9B370
// Address: 0x1b9b370
//
__int64 __fastcall sub_1B9B370(__int64 a1, __int64 *a2, __int64 *a3, __m128i a4, __m128i a5, double a6)
{
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rdx
  __int64 *v14; // r9
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  unsigned int v17; // r13d
  unsigned __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r11
  bool v22; // zf
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rax
  unsigned int v27; // r13d
  __int64 v28; // r8
  unsigned int *v29; // r8
  int v30; // r9d
  unsigned int *v31; // r8
  _BYTE *v33; // r9
  size_t v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  int v39; // edx
  int v40; // r10d
  char v41; // [rsp+1Fh] [rbp-241h]
  __int64 v42; // [rsp+20h] [rbp-240h]
  char v43; // [rsp+28h] [rbp-238h]
  __int64 v44; // [rsp+30h] [rbp-230h]
  __int64 v45; // [rsp+30h] [rbp-230h]
  __int64 v46; // [rsp+30h] [rbp-230h]
  __int64 v47; // [rsp+38h] [rbp-228h]
  unsigned __int64 v48; // [rsp+38h] [rbp-228h]
  unsigned __int64 v49[3]; // [rsp+40h] [rbp-220h] BYREF
  int v50; // [rsp+58h] [rbp-208h]
  __int64 v51; // [rsp+60h] [rbp-200h]
  __int64 v52; // [rsp+68h] [rbp-1F8h]
  _BYTE *v53; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v54; // [rsp+78h] [rbp-1E8h]
  _BYTE v55[16]; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v56[2]; // [rsp+90h] [rbp-1D0h] BYREF
  const char *v57; // [rsp+A0h] [rbp-1C0h]
  __int64 v58; // [rsp+A8h] [rbp-1B8h]
  __int64 v59; // [rsp+B0h] [rbp-1B0h]
  __int64 v60; // [rsp+B8h] [rbp-1A8h]
  int v61; // [rsp+C0h] [rbp-1A0h]
  __int64 v62; // [rsp+C8h] [rbp-198h]
  __int64 v63; // [rsp+D0h] [rbp-190h]
  __int64 v64; // [rsp+D8h] [rbp-188h]
  __int64 v65; // [rsp+E0h] [rbp-180h]
  __int64 v66; // [rsp+E8h] [rbp-178h]
  __int64 v67; // [rsp+F0h] [rbp-170h]
  __int64 v68; // [rsp+F8h] [rbp-168h]
  __int64 v69; // [rsp+100h] [rbp-160h]
  __int64 v70; // [rsp+108h] [rbp-158h]
  __int64 v71; // [rsp+110h] [rbp-150h]
  __int64 v72; // [rsp+118h] [rbp-148h]
  int v73; // [rsp+120h] [rbp-140h]
  __int64 v74; // [rsp+128h] [rbp-138h]
  _BYTE *v75; // [rsp+130h] [rbp-130h]
  _BYTE *v76; // [rsp+138h] [rbp-128h]
  __int64 v77; // [rsp+140h] [rbp-120h]
  int v78; // [rsp+148h] [rbp-118h]
  _BYTE v79[16]; // [rsp+150h] [rbp-110h] BYREF
  __int64 v80; // [rsp+160h] [rbp-100h]
  __int64 v81; // [rsp+168h] [rbp-F8h]
  __int64 v82; // [rsp+170h] [rbp-F0h]
  __int64 v83; // [rsp+178h] [rbp-E8h]
  __int64 v84; // [rsp+180h] [rbp-E0h]
  __int64 v85; // [rsp+188h] [rbp-D8h]
  __int16 v86; // [rsp+190h] [rbp-D0h]
  __int64 v87; // [rsp+198h] [rbp-C8h]
  __int64 v88; // [rsp+1A0h] [rbp-C0h]
  __int64 v89; // [rsp+1A8h] [rbp-B8h]
  __int64 v90; // [rsp+1B0h] [rbp-B0h]
  __int64 v91; // [rsp+1B8h] [rbp-A8h]
  int v92; // [rsp+1C0h] [rbp-A0h]
  __int64 v93; // [rsp+1C8h] [rbp-98h]
  __int64 v94; // [rsp+1D0h] [rbp-90h]
  __int64 v95; // [rsp+1D8h] [rbp-88h]
  char *v96; // [rsp+1E0h] [rbp-80h]
  __int64 v97; // [rsp+1E8h] [rbp-78h]
  char v98; // [rsp+1F0h] [rbp-70h] BYREF

  v9 = *(_QWORD *)(a1 + 448);
  v10 = *(unsigned int *)(v9 + 128);
  if ( !(_DWORD)v10 )
    goto LABEL_48;
  v11 = *(_QWORD *)(v9 + 112);
  v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = v11 + 16LL * v12;
  v14 = *(__int64 **)v13;
  if ( *(__int64 **)v13 != a2 )
  {
    v39 = 1;
    while ( v14 != (__int64 *)-8LL )
    {
      v40 = v39 + 1;
      v12 = (v10 - 1) & (v39 + v12);
      v13 = v11 + 16LL * v12;
      v14 = *(__int64 **)v13;
      if ( *(__int64 **)v13 == a2 )
        goto LABEL_3;
      v39 = v40;
    }
    goto LABEL_48;
  }
LABEL_3:
  if ( v13 == v11 + 16 * v10 )
  {
LABEL_48:
    v15 = *(_QWORD *)(v9 + 144);
    goto LABEL_5;
  }
  v15 = *(_QWORD *)(v9 + 136) + 88LL * *(unsigned int *)(v13 + 8);
LABEL_5:
  v49[0] = 6;
  v16 = *(_QWORD *)(v15 + 24);
  v49[1] = 0;
  v49[2] = v16;
  if ( v16 != 0 && v16 != -8 && v16 != -16 )
    sub_1649AC0(v49, *(_QWORD *)(v15 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  v17 = *(_DWORD *)(v15 + 64);
  v50 = *(_DWORD *)(v15 + 32);
  v51 = *(_QWORD *)(v15 + 40);
  v52 = *(_QWORD *)(v15 + 48);
  v53 = v55;
  v54 = 0x200000000LL;
  if ( v17 && &v53 != (_BYTE **)(v15 + 56) )
  {
    v33 = v55;
    v34 = 8LL * v17;
    if ( v17 <= 2
      || (sub_16CD150((__int64)&v53, v55, v17, 8, v17, (int)v55),
          v33 = v53,
          (v34 = 8LL * *(unsigned int *)(v15 + 64)) != 0) )
    {
      memcpy(v33, *(const void **)(v15 + 56), v34);
    }
    LODWORD(v54) = v17;
  }
  v18 = (unsigned __int64)a2;
  v41 = 0;
  if ( a3 )
    v18 = (unsigned __int64)a3;
  if ( *(_DWORD *)(a1 + 88) > 1u )
    v41 = sub_1B97AA0(a1, v18);
  v19 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
  v44 = sub_1632FA0(v19);
  if ( sub_1456C80(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL), *a2) )
  {
    v35 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
    v57 = "induction";
    v75 = v79;
    v76 = v79;
    v56[1] = v44;
    v56[0] = v35;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v77 = 2;
    v78 = 0;
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = 1;
    v36 = sub_15E0530(*(_QWORD *)(v35 + 24));
    v37 = *(_QWORD *)(a1 + 168);
    v87 = 0;
    v90 = v36;
    v96 = &v98;
    v95 = v44;
    v89 = 0;
    v91 = 0;
    v92 = 0;
    v93 = 0;
    v94 = 0;
    v88 = 0;
    v97 = 0x800000000LL;
    v48 = sub_157EBA0(v37);
    v38 = sub_1456040(v51);
    v47 = sub_38767A0(v56, v51, v38, v48);
    sub_194A930((__int64)v56);
  }
  else
  {
    if ( !v51 )
      BUG();
    v47 = *(_QWORD *)(v51 - 8);
  }
  if ( *(_DWORD *)(a1 + 88) <= 1u || (unsigned __int8)sub_1B97910(a1, v18) )
  {
    v43 = 0;
  }
  else
  {
    sub_1B9A8D0(a1, (__int64)v49, v47, v18, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
    if ( !v41 )
      goto LABEL_35;
    v43 = v41;
  }
  v20 = *(_QWORD *)(a1 + 264);
  v42 = v20;
  if ( *(__int64 **)(a1 + 272) != a2 )
  {
    v21 = *a2;
    v22 = *(_BYTE *)(*a2 + 8) == 11;
    LOWORD(v57) = 257;
    if ( v22 )
      v23 = sub_1904CF0((__int64 *)(a1 + 96), v20, v21, v56);
    else
      v23 = sub_12AA3B0((__int64 *)(a1 + 96), 0x2Au, v20, v21, (__int64)v56);
    v24 = sub_1B19340((__int64)v49, a1 + 96, v23, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 112LL), v44, a4, a5, a6);
    v56[0] = (__int64)"offset.idx";
    v42 = (__int64)v24;
    LOWORD(v57) = 259;
    sub_164B780((__int64)v24, v56);
  }
  if ( a3 )
  {
    v25 = *a3;
    LOWORD(v57) = 257;
    v45 = v25;
    v26 = sub_12AA3B0((__int64 *)(a1 + 96), 0x24u, v42, v25, (__int64)v56);
    LOWORD(v57) = 257;
    v42 = v26;
    v47 = sub_12AA3B0((__int64 *)(a1 + 96), 0x24u, v47, v45, (__int64)v56);
  }
  if ( !v43 )
  {
    v46 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 16LL))(a1, v42);
    if ( *(_DWORD *)(a1 + 92) )
    {
      v27 = 0;
      do
      {
        if ( v52 )
          v28 = (unsigned int)*(unsigned __int8 *)(v52 + 16) - 24;
        else
          v28 = 29;
        v56[0] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)a1 + 24LL))(
                   a1,
                   v46,
                   v27 * *(_DWORD *)(a1 + 88),
                   v47,
                   v28);
        sub_1B99BD0((unsigned int *)(a1 + 280), v18, v27, v56[0], v29, v30);
        if ( a3 )
          sub_1B916B0(a1, v56, 1, (__int64)a3);
        v31 = (unsigned int *)v27++;
        sub_1B9A880(a1, (__int64)v49, v18, v56[0], v31, (unsigned __int64 *)0xFFFFFFFFLL);
      }
      while ( *(_DWORD *)(a1 + 92) > v27 );
    }
  }
  if ( v41 )
    sub_1B9AF00(a1, v42, v47, v18, (__int64)v49, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
LABEL_35:
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return sub_1455FA0((__int64)v49);
}
