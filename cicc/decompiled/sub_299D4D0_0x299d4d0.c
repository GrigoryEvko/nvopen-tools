// Function: sub_299D4D0
// Address: 0x299d4d0
//
void __fastcall sub_299D4D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 *v30; // rbx
  unsigned __int64 *v31; // r12
  unsigned __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // edx
  unsigned int v39; // ebx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r12
  unsigned __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  unsigned __int64 *v50; // rbx
  unsigned __int64 *v51; // r12
  unsigned __int64 v52; // rdi
  unsigned __int64 *v53; // rbx
  unsigned __int64 v54; // rdi
  __int64 v55; // [rsp+8h] [rbp-218h]
  __int64 v56; // [rsp+8h] [rbp-218h]
  __int64 v57; // [rsp+8h] [rbp-218h]
  __int64 v58; // [rsp+8h] [rbp-218h]
  __int64 v59; // [rsp+8h] [rbp-218h]
  __int64 *v60; // [rsp+10h] [rbp-210h] BYREF
  __int64 v61; // [rsp+18h] [rbp-208h]
  __int64 *v62; // [rsp+20h] [rbp-200h] BYREF
  int v63; // [rsp+28h] [rbp-1F8h]
  __m128i v64; // [rsp+30h] [rbp-1F0h] BYREF
  _QWORD v65[10]; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int64 *v66; // [rsp+90h] [rbp-190h]
  unsigned int v67; // [rsp+98h] [rbp-188h]
  _BYTE v68[384]; // [rsp+A0h] [rbp-180h] BYREF

  v6 = a2;
  if ( (unsigned int)sub_F6E5D0(a1, (__int64)a2, a3, a4, a5, a6) != 5 )
    goto LABEL_2;
  v55 = **(_QWORD **)(a1 + 32);
  sub_D4BD20(&v62, a1, v8, v9, v10, v55);
  sub_B157E0((__int64)&v64, &v62);
  sub_B17AD0((__int64)v65, (__int64)"transform-warning", (__int64)"FailedRequestedUnrolling", 24, &v64, v55);
  sub_B18290(
    (__int64)v65,
    "loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation might be dis"
    "abled or specified as part of an unsupported transformation ordering",
    0xB8u);
  sub_1049740(a2, (__int64)v65);
  v23 = v66;
  v65[0] = &unk_49D9D40;
  v24 = &v66[10 * v67];
  if ( v66 != v24 )
  {
    do
    {
      v24 -= 10;
      v25 = v24[4];
      if ( (unsigned __int64 *)v25 != v24 + 6 )
        j_j___libc_free_0(v25);
      if ( (unsigned __int64 *)*v24 != v24 + 2 )
        j_j___libc_free_0(*v24);
    }
    while ( v23 != v24 );
    v24 = v66;
  }
  if ( v24 != (unsigned __int64 *)v68 )
    _libc_free((unsigned __int64)v24);
  a2 = v62;
  if ( v62 )
  {
    sub_B91220((__int64)&v62, (__int64)v62);
    if ( (unsigned int)sub_F6E690(a1, (__int64)a2, v26, v27, v28, v29) != 5 )
      goto LABEL_3;
  }
  else
  {
LABEL_2:
    if ( (unsigned int)sub_F6E690(a1, (__int64)a2, v8, v9, v10, v11) != 5 )
      goto LABEL_3;
  }
  v56 = **(_QWORD **)(a1 + 32);
  sub_D4BD20(&v62, a1, v12, v13, v14, v56);
  sub_B157E0((__int64)&v64, &v62);
  sub_B17AD0((__int64)v65, (__int64)"transform-warning", (__int64)"FailedRequestedUnrollAndJamming", 31, &v64, v56);
  sub_B18290(
    (__int64)v65,
    "loop not unroll-and-jammed: the optimizer was unable to perform the requested transformation; the transformation mig"
    "ht be disabled or specified as part of an unsupported transformation ordering",
    0xC1u);
  sub_1049740(v6, (__int64)v65);
  v30 = v66;
  v65[0] = &unk_49D9D40;
  v31 = &v66[10 * v67];
  if ( v66 != v31 )
  {
    do
    {
      v31 -= 10;
      v32 = v31[4];
      if ( (unsigned __int64 *)v32 != v31 + 6 )
        j_j___libc_free_0(v32);
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
    }
    while ( v30 != v31 );
    v31 = v66;
  }
  if ( v31 != (unsigned __int64 *)v68 )
    _libc_free((unsigned __int64)v31);
  a2 = v62;
  if ( !v62 )
  {
LABEL_3:
    if ( (unsigned int)sub_F6E730(a1, (__int64)a2, v12, v13, v14, v15) != 5 )
      goto LABEL_4;
    goto LABEL_28;
  }
  sub_B91220((__int64)&v62, (__int64)v62);
  if ( (unsigned int)sub_F6E730(a1, (__int64)a2, v33, v34, v35, v36) != 5 )
    goto LABEL_4;
LABEL_28:
  v37 = sub_F6E040(a1, (__int64)a2, v16, v17, v18, v19);
  a2 = (__int64 *)"llvm.loop.interleave.count";
  v63 = v38;
  v39 = v37;
  v62 = (__int64 *)v37;
  v61 = sub_D4A2B0(a1, "llvm.loop.interleave.count", 0x1Au, v40, v41, v42);
  if ( (_BYTE)v63 )
  {
    if ( BYTE4(v62) )
    {
      if ( !v39 )
      {
LABEL_31:
        if ( BYTE4(v61) && (_DWORD)v61 == 1 )
          goto LABEL_4;
        v57 = **(_QWORD **)(a1 + 32);
        sub_D4BD20(&v60, a1, v16, v17, v18, v57);
        sub_B157E0((__int64)&v64, &v60);
        sub_B17AD0((__int64)v65, (__int64)"transform-warning", (__int64)"FailedRequestedInterleaving", 27, &v64, v57);
        sub_B18290(
          (__int64)v65,
          "loop not interleaved: the optimizer was unable to perform the requested transformation; the transformation mig"
          "ht be disabled or specified as part of an unsupported transformation ordering",
          0xBBu);
        sub_1049740(v6, (__int64)v65);
        v43 = v66;
        v65[0] = &unk_49D9D40;
        v44 = &v66[10 * v67];
        if ( v66 != v44 )
        {
          do
          {
            v44 -= 10;
            v45 = v44[4];
            if ( (unsigned __int64 *)v45 != v44 + 6 )
              j_j___libc_free_0(v45);
            if ( (unsigned __int64 *)*v44 != v44 + 2 )
              j_j___libc_free_0(*v44);
          }
          while ( v43 != v44 );
LABEL_39:
          v44 = v66;
          goto LABEL_40;
        }
        goto LABEL_40;
      }
    }
    else if ( v39 <= 1 )
    {
      goto LABEL_31;
    }
  }
  v59 = **(_QWORD **)(a1 + 32);
  sub_D4BD20(&v60, a1, v16, v17, v18, v59);
  sub_B157E0((__int64)&v64, &v60);
  sub_B17AD0((__int64)v65, (__int64)"transform-warning", (__int64)"FailedRequestedVectorization", 28, &v64, v59);
  sub_B18290(
    (__int64)v65,
    "loop not vectorized: the optimizer was unable to perform the requested transformation; the transformation might be d"
    "isabled or specified as part of an unsupported transformation ordering",
    0xBAu);
  sub_1049740(v6, (__int64)v65);
  v53 = v66;
  v65[0] = &unk_49D9D40;
  v44 = &v66[10 * v67];
  if ( v66 != v44 )
  {
    do
    {
      v44 -= 10;
      v54 = v44[4];
      if ( (unsigned __int64 *)v54 != v44 + 6 )
        j_j___libc_free_0(v54);
      if ( (unsigned __int64 *)*v44 != v44 + 2 )
        j_j___libc_free_0(*v44);
    }
    while ( v53 != v44 );
    goto LABEL_39;
  }
LABEL_40:
  if ( v44 != (unsigned __int64 *)v68 )
    _libc_free((unsigned __int64)v44);
  a2 = v60;
  if ( v60 )
  {
    sub_B91220((__int64)&v60, (__int64)v60);
    if ( (unsigned int)sub_F6E900(a1, (__int64)a2, v46, v47, v48, v49) != 5 )
      return;
    goto LABEL_44;
  }
LABEL_4:
  if ( (unsigned int)sub_F6E900(a1, (__int64)a2, v16, v17, v18, v19) != 5 )
    return;
LABEL_44:
  v58 = **(_QWORD **)(a1 + 32);
  sub_D4BD20(&v62, a1, v20, v21, v22, v58);
  sub_B157E0((__int64)&v64, &v62);
  sub_B17AD0((__int64)v65, (__int64)"transform-warning", (__int64)"FailedRequestedDistribution", 27, &v64, v58);
  sub_B18290(
    (__int64)v65,
    "loop not distributed: the optimizer was unable to perform the requested transformation; the transformation might be "
    "disabled or specified as part of an unsupported transformation ordering",
    0xBBu);
  sub_1049740(v6, (__int64)v65);
  v50 = v66;
  v65[0] = &unk_49D9D40;
  v51 = &v66[10 * v67];
  if ( v66 != v51 )
  {
    do
    {
      v51 -= 10;
      v52 = v51[4];
      if ( (unsigned __int64 *)v52 != v51 + 6 )
        j_j___libc_free_0(v52);
      if ( (unsigned __int64 *)*v51 != v51 + 2 )
        j_j___libc_free_0(*v51);
    }
    while ( v50 != v51 );
    v51 = v66;
  }
  if ( v51 != (unsigned __int64 *)v68 )
    _libc_free((unsigned __int64)v51);
  if ( v62 )
    sub_B91220((__int64)&v62, (__int64)v62);
}
