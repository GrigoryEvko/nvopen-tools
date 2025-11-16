// Function: sub_1BAD7F0
// Address: 0x1bad7f0
//
__int64 __fastcall sub_1BAD7F0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v12; // r12
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned int v22; // r14d
  unsigned __int64 v23; // rax
  __int64 v24; // r14
  _QWORD *v25; // rax
  _QWORD *v26; // r9
  unsigned __int64 v27; // rax
  double v28; // xmm4_8
  double v29; // xmm5_8
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rax
  _QWORD *v35; // [rsp+8h] [rbp-1F8h]
  _QWORD *v36; // [rsp+8h] [rbp-1F8h]
  __int64 v37[2]; // [rsp+10h] [rbp-1F0h] BYREF
  char v38; // [rsp+20h] [rbp-1E0h]
  char v39; // [rsp+21h] [rbp-1DFh]
  _QWORD v40[6]; // [rsp+30h] [rbp-1D0h] BYREF
  int v41; // [rsp+60h] [rbp-1A0h]
  __int64 v42; // [rsp+68h] [rbp-198h]
  __int64 v43; // [rsp+70h] [rbp-190h]
  __int64 v44; // [rsp+78h] [rbp-188h]
  __int64 v45; // [rsp+80h] [rbp-180h]
  __int64 v46; // [rsp+88h] [rbp-178h]
  __int64 v47; // [rsp+90h] [rbp-170h]
  __int64 v48; // [rsp+98h] [rbp-168h]
  __int64 v49; // [rsp+A0h] [rbp-160h]
  __int64 v50; // [rsp+A8h] [rbp-158h]
  __int64 v51; // [rsp+B0h] [rbp-150h]
  __int64 v52; // [rsp+B8h] [rbp-148h]
  int v53; // [rsp+C0h] [rbp-140h]
  __int64 v54; // [rsp+C8h] [rbp-138h]
  _BYTE *v55; // [rsp+D0h] [rbp-130h]
  _BYTE *v56; // [rsp+D8h] [rbp-128h]
  __int64 v57; // [rsp+E0h] [rbp-120h]
  int v58; // [rsp+E8h] [rbp-118h]
  _BYTE v59[16]; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v60; // [rsp+100h] [rbp-100h]
  __int64 v61; // [rsp+108h] [rbp-F8h]
  __int64 v62; // [rsp+110h] [rbp-F0h]
  __int64 v63; // [rsp+118h] [rbp-E8h]
  __int64 v64; // [rsp+120h] [rbp-E0h]
  __int64 v65; // [rsp+128h] [rbp-D8h]
  __int16 v66; // [rsp+130h] [rbp-D0h]
  __int64 v67; // [rsp+138h] [rbp-C8h]
  __int64 v68; // [rsp+140h] [rbp-C0h]
  __int64 v69; // [rsp+148h] [rbp-B8h]
  __int64 v70; // [rsp+150h] [rbp-B0h]
  __int64 v71; // [rsp+158h] [rbp-A8h]
  int v72; // [rsp+160h] [rbp-A0h]
  __int64 v73; // [rsp+168h] [rbp-98h]
  __int64 v74; // [rsp+170h] [rbp-90h]
  __int64 v75; // [rsp+178h] [rbp-88h]
  char *v76; // [rsp+180h] [rbp-80h]
  __int64 v77; // [rsp+188h] [rbp-78h]
  char v78; // [rsp+190h] [rbp-70h] BYREF

  v12 = (_QWORD *)sub_13FC520((__int64)a2);
  v13 = sub_157EB90(a3);
  memset(&v40[3], 0, 24);
  v14 = sub_1632FA0(v13);
  v15 = *(_QWORD *)(a1 + 16);
  v40[2] = "scev.check";
  v40[1] = v14;
  v16 = *(_QWORD *)(v15 + 112);
  v55 = v59;
  v56 = v59;
  v40[0] = v16;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v57 = 2;
  v58 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 1;
  v17 = sub_15E0530(*(_QWORD *)(v16 + 24));
  v75 = v14;
  v70 = v17;
  v76 = &v78;
  v67 = 0;
  v69 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v68 = 0;
  v77 = 0x800000000LL;
  v18 = sub_157EBA0((__int64)v12);
  v19 = sub_1458800(*(_QWORD *)(a1 + 16));
  v20 = sub_387DD00(v40, v19, v18);
  v21 = v20;
  if ( *(_BYTE *)(v20 + 16) == 13 )
  {
    v22 = *(_DWORD *)(v20 + 32);
    if ( v22 <= 0x40 )
    {
      if ( !*(_QWORD *)(v20 + 24) )
        return sub_194A930((__int64)v40);
    }
    else if ( v22 == (unsigned int)sub_16A57B0(v20 + 24) )
    {
      return sub_194A930((__int64)v40);
    }
  }
  v39 = 1;
  v37[0] = (__int64)"vector.scevcheck";
  v38 = 3;
  sub_164B780((__int64)v12, v37);
  v39 = 1;
  v37[0] = (__int64)"vector.ph";
  v38 = 3;
  v23 = sub_157EBA0((__int64)v12);
  v24 = sub_157FBF0(v12, (__int64 *)(v23 + 24), (__int64)v37);
  sub_1BACEB0(*(_QWORD *)(a1 + 32), v24, (__int64)v12);
  if ( *a2 )
    sub_1400330(*a2, v24, *(_QWORD *)(a1 + 24));
  v25 = sub_1648A60(56, 3u);
  v26 = v25;
  if ( v25 )
  {
    v35 = v25;
    sub_15F83E0((__int64)v25, a3, v24, v21, 0);
    v26 = v35;
  }
  v36 = v26;
  v27 = sub_157EBA0((__int64)v12);
  sub_1AA6530(v27, v36, a4, a5, a6, a7, v28, v29, a10, a11);
  v32 = *(unsigned int *)(a1 + 224);
  if ( (unsigned int)v32 >= *(_DWORD *)(a1 + 228) )
  {
    sub_16CD150(a1 + 216, (const void *)(a1 + 232), 0, 8, v30, v31);
    v32 = *(unsigned int *)(a1 + 224);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v32) = v12;
  ++*(_DWORD *)(a1 + 224);
  *(_BYTE *)(a1 + 464) = 1;
  return sub_194A930((__int64)v40);
}
