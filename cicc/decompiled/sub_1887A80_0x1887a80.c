// Function: sub_1887A80
// Address: 0x1887a80
//
void __fastcall sub_1887A80(
        __int64 **a1,
        __int64 a2,
        __int64 *a3,
        char a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 *v12; // rax
  __int64 *v14; // rbx
  __int64 v15; // rdi
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int8 *v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // r10
  _QWORD *v21; // rax
  _QWORD *v22; // r13
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  unsigned __int8 *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // r12
  __int64 *v33; // r8
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r9
  __int64 v40; // rsi
  __int64 v41; // r13
  _QWORD *v42; // rdi
  const char *v43; // r13
  size_t v44; // rax
  __int64 *v45; // rbx
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r13
  __int64 v52; // rdx
  __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 *v62; // [rsp+28h] [rbp-168h]
  __int64 v63; // [rsp+30h] [rbp-160h]
  unsigned __int64 *v64; // [rsp+30h] [rbp-160h]
  __int64 v65; // [rsp+30h] [rbp-160h]
  unsigned int v66; // [rsp+38h] [rbp-158h]
  __int64 v67; // [rsp+38h] [rbp-158h]
  __int64 v68; // [rsp+38h] [rbp-158h]
  _QWORD *v69; // [rsp+38h] [rbp-158h]
  __int64 v70; // [rsp+38h] [rbp-158h]
  __int64 v71; // [rsp+38h] [rbp-158h]
  char *v72; // [rsp+48h] [rbp-148h] BYREF
  unsigned __int8 *v73[2]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v74; // [rsp+60h] [rbp-130h]
  char *v75; // [rsp+70h] [rbp-120h] BYREF
  __int64 v76; // [rsp+78h] [rbp-118h]
  unsigned __int64 *v77; // [rsp+80h] [rbp-110h]
  __int64 v78; // [rsp+88h] [rbp-108h]
  __int64 v79; // [rsp+90h] [rbp-100h]
  int v80; // [rsp+98h] [rbp-F8h]
  __int64 v81; // [rsp+A0h] [rbp-F0h]
  __int64 v82; // [rsp+A8h] [rbp-E8h]
  __int64 v83; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+C8h] [rbp-C8h]
  __int64 v85; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 *v86; // [rsp+110h] [rbp-80h] BYREF
  __int64 v87; // [rsp+118h] [rbp-78h]
  _BYTE v88[112]; // [rsp+120h] [rbp-70h] BYREF

  v12 = (unsigned __int64 *)&v85;
  v83 = 0;
  v84 = 1;
  do
    *v12++ = -8;
  while ( v12 != (unsigned __int64 *)&v86 );
  v86 = (__int64 *)v88;
  v87 = 0x800000000LL;
  sub_18870B0(a2, (__int64)&v83);
  v14 = v86;
  v62 = &v86[(unsigned int)v87];
  if ( v86 != v62 )
  {
    do
    {
      v31 = a1[21];
      v32 = *v14;
      if ( !v31 )
      {
        v33 = *a1;
        v75 = "__cfi_global_var_init";
        LOWORD(v77) = 259;
        v65 = (__int64)v33;
        v34 = (__int64 *)sub_1643270((_QWORD *)*v33);
        v67 = sub_16453E0(v34, 0);
        v35 = sub_1648B60(120);
        v36 = v35;
        if ( v35 )
          sub_15E2490(v35, v67, 7, (__int64)&v75, v65);
        a1[21] = (__int64 *)v36;
        v75 = "entry";
        v37 = *a1;
        LOWORD(v77) = 259;
        v68 = *v37;
        v38 = (_QWORD *)sub_22077B0(64);
        v39 = v38;
        if ( v38 )
        {
          v40 = v68;
          v69 = v38;
          sub_157FB60(v38, v40, (__int64)&v75, v36, 0);
          v39 = v69;
        }
        v70 = (__int64)v39;
        v41 = **a1;
        v42 = sub_1648A60(56, 0);
        if ( v42 )
          sub_15F7190((__int64)v42, v41, v70);
        v43 = ".text.startup";
        if ( *((_DWORD *)a1 + 8) == 3 )
          v43 = "__TEXT,__StaticInit,regular,pure_instructions";
        v71 = (__int64)a1[21];
        v44 = strlen(v43);
        sub_15E5D20(v71, v43, v44);
        sub_1B28000(*a1, a1[21], 0, 0);
        v31 = a1[21];
      }
      v15 = v31[10];
      if ( v15 )
        v15 -= 24;
      v16 = sub_157EBA0(v15);
      v17 = sub_16498A0(v16);
      v75 = 0;
      v78 = v17;
      v79 = 0;
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v76 = *(_QWORD *)(v16 + 40);
      v77 = (unsigned __int64 *)(v16 + 24);
      v18 = *(unsigned __int8 **)(v16 + 48);
      v73[0] = v18;
      if ( v18 )
      {
        sub_1623A60((__int64)v73, (__int64)v18, 2);
        if ( v75 )
          sub_161E7C0((__int64)&v75, (__int64)v75);
        v75 = (char *)v73[0];
        if ( v73[0] )
          sub_1623210((__int64)v73, v73[0], (__int64)&v75);
      }
      v19 = *(_DWORD *)(v32 + 32);
      *(_BYTE *)(v32 + 80) &= ~1u;
      v20 = *(_QWORD *)(v32 - 24);
      v74 = 257;
      v63 = v20;
      v66 = (unsigned int)(1 << (v19 >> 15)) >> 1;
      v21 = sub_1648A60(64, 2u);
      v22 = v21;
      if ( v21 )
        sub_15F9650((__int64)v21, v63, v32, 0, 0);
      if ( v76 )
      {
        v64 = v77;
        sub_157E9D0(v76 + 40, (__int64)v22);
        v23 = *v64;
        v24 = v22[3] & 7LL;
        v22[4] = v64;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        v22[3] = v23 | v24;
        *(_QWORD *)(v23 + 8) = v22 + 3;
        *v64 = *v64 & 7 | (unsigned __int64)(v22 + 3);
      }
      sub_164B780((__int64)v22, (__int64 *)v73);
      if ( v75 )
      {
        v72 = v75;
        sub_1623A60((__int64)&v72, (__int64)v75, 2);
        v25 = v22[6];
        v26 = (__int64)(v22 + 6);
        if ( v25 )
        {
          sub_161E7C0((__int64)(v22 + 6), v25);
          v26 = (__int64)(v22 + 6);
        }
        v27 = (unsigned __int8 *)v72;
        v22[6] = v72;
        if ( v27 )
          sub_1623210((__int64)&v72, v27, v26);
      }
      sub_15F9450((__int64)v22, v66);
      v30 = sub_15A06D0(*(__int64 ***)(v32 + 24), v66, v28, v29);
      sub_15E5440(v32, v30);
      if ( v75 )
        sub_161E7C0((__int64)&v75, (__int64)v75);
      ++v14;
    }
    while ( v62 != v14 );
  }
  v45 = *a1;
  LOWORD(v77) = 257;
  v46 = *(_QWORD *)(a2 + 24);
  v47 = sub_1648B60(120);
  v48 = v47;
  if ( v47 )
    sub_15E2490(v47, v46, 9, (__int64)&v75, (__int64)v45);
  sub_1887680(a2, v48, a4, *(double *)a5.m128_u64, a6, a7);
  v51 = sub_15A06D0(*(__int64 ***)a2, v48, v49, v50);
  v54 = (_QWORD *)sub_15A06D0(*(__int64 ***)a2, v48, v52, v53);
  v55 = sub_15A35F0(0x21u, (_QWORD *)a2, v54, 0);
  v56 = sub_15A2DC0(v55, a3, v51, 0);
  sub_164D160(v48, v56, a5, a6, a7, a8, v57, v58, a11, a12);
  sub_15E3D00(v48);
  if ( v86 != (__int64 *)v88 )
    _libc_free((unsigned __int64)v86);
  if ( (v84 & 1) == 0 )
    j___libc_free_0(v85);
}
