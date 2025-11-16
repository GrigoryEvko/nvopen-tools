// Function: sub_1B3B3D0
// Address: 0x1b3b3d0
//
void __fastcall sub_1B3B3D0(
        void *src,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        char a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v13; // r12
  char *v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  _QWORD *v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // rax
  unsigned __int64 *v27; // rax
  unsigned __int64 *v28; // r12
  char *v29; // [rsp-3F8h] [rbp-3F8h] BYREF
  char *v30; // [rsp-3F0h] [rbp-3F0h]
  char *v31; // [rsp-3E8h] [rbp-3E8h]
  _QWORD *v32; // [rsp-3E0h] [rbp-3E0h]
  _QWORD v33[64]; // [rsp-3D8h] [rbp-3D8h] BYREF
  char v34; // [rsp-1D8h] [rbp-1D8h]
  __int64 v35; // [rsp-1D0h] [rbp-1D0h]
  __int64 v36; // [rsp-1C8h] [rbp-1C8h]
  __int64 v37; // [rsp-1C0h] [rbp-1C0h]
  int v38; // [rsp-1B8h] [rbp-1B8h]
  __int64 v39; // [rsp-1B0h] [rbp-1B0h]
  __int64 v40; // [rsp-1A8h] [rbp-1A8h]
  __int64 v41; // [rsp-1A0h] [rbp-1A0h]
  int v42; // [rsp-198h] [rbp-198h]
  __int64 v43; // [rsp-190h] [rbp-190h]
  __int64 v44; // [rsp-188h] [rbp-188h]
  __int64 v45; // [rsp-180h] [rbp-180h]
  int v46; // [rsp-178h] [rbp-178h]
  _QWORD *v47; // [rsp-170h] [rbp-170h]
  __int64 v48; // [rsp-168h] [rbp-168h]
  _QWORD v49[9]; // [rsp-160h] [rbp-160h] BYREF
  _QWORD *v50; // [rsp-118h] [rbp-118h]
  _QWORD *v51; // [rsp-110h] [rbp-110h]
  __int64 v52; // [rsp-108h] [rbp-108h]
  int v53; // [rsp-100h] [rbp-100h]
  _QWORD v54[17]; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v55; // [rsp-70h] [rbp-70h]
  __int64 v56; // [rsp-68h] [rbp-68h]
  int v57; // [rsp-60h] [rbp-60h]
  __int64 v58; // [rsp-58h] [rbp-58h]
  __int64 v59; // [rsp-50h] [rbp-50h]
  __int64 v60; // [rsp-48h] [rbp-48h]
  int v61; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v13 = 8 * a2;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    if ( (unsigned __int64)(8 * a2) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v17 = 0;
    if ( v13 )
    {
      v29 = (char *)sub_22077B0(8 * a2);
      v31 = &v29[v13];
      memcpy(v29, src, 8 * a2);
      v17 = &v29[v13];
    }
    v18 = *a3;
    v30 = v17;
    v32 = a3;
    v19 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v18 + 56LL) + 40LL);
    sub_15A5590((__int64)v33, (__int64 *)v19, 0, 0);
    v20 = *a3;
    v33[58] = a4;
    v21 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v20 + 56LL) + 40LL));
    v34 = a5;
    v33[59] = v21;
    v48 = 0x800000000LL;
    v50 = v54;
    v51 = v54;
    v33[60] = 0;
    v33[61] = a3;
    v33[62] = a4;
    v33[63] = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = v49;
    v49[8] = 0;
    v52 = 16;
    v53 = 0;
    v54[16] = 0;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    sub_1B382B0((__int64)&v29, a6, a7, a8, a9, v22, v23, a12, a13);
    j___libc_free_0(v59);
    j___libc_free_0(v55);
    if ( v51 != v50 )
      _libc_free((unsigned __int64)v51);
    v24 = v47;
    v25 = &v47[(unsigned int)v48];
    if ( v47 != v25 )
    {
      do
      {
        v26 = *--v25;
        if ( (v26 & 4) != 0 )
        {
          v27 = (unsigned __int64 *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
          v28 = v27;
          if ( v27 )
          {
            if ( (unsigned __int64 *)*v27 != v27 + 2 )
              _libc_free(*v27);
            v19 = 48;
            j_j___libc_free_0(v28, 48);
          }
        }
      }
      while ( v24 != v25 );
      v25 = v47;
    }
    if ( v25 != v49 )
      _libc_free((unsigned __int64)v25);
    j___libc_free_0(v44);
    j___libc_free_0(v40);
    j___libc_free_0(v36);
    sub_129E320((__int64)v33, v19);
    if ( v29 )
      j_j___libc_free_0(v29, v31 - v29);
  }
}
