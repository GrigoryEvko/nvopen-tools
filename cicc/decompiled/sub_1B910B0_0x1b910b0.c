// Function: sub_1B910B0
// Address: 0x1b910b0
//
__int64 __fastcall sub_1B910B0(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // rdi
  const char **v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // [rsp+10h] [rbp-250h]
  __int64 v29; // [rsp+18h] [rbp-248h]
  __int64 v30; // [rsp+18h] [rbp-248h]
  __int64 *v31; // [rsp+18h] [rbp-248h]
  const char *v32; // [rsp+20h] [rbp-240h] BYREF
  char v33; // [rsp+30h] [rbp-230h]
  char v34; // [rsp+31h] [rbp-22Fh]
  __int64 v35[5]; // [rsp+40h] [rbp-220h] BYREF
  int v36; // [rsp+68h] [rbp-1F8h]
  __int64 v37; // [rsp+70h] [rbp-1F0h]
  __int64 v38; // [rsp+78h] [rbp-1E8h]
  const char **v39; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v40; // [rsp+98h] [rbp-1C8h]
  const char *v41; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-1B8h]
  __int64 v43; // [rsp+B0h] [rbp-1B0h]
  __int64 v44; // [rsp+B8h] [rbp-1A8h]
  int v45; // [rsp+C0h] [rbp-1A0h]
  __int64 v46; // [rsp+C8h] [rbp-198h]
  __int64 v47; // [rsp+D0h] [rbp-190h]
  __int64 v48; // [rsp+D8h] [rbp-188h]
  __int64 v49; // [rsp+E0h] [rbp-180h]
  __int64 v50; // [rsp+E8h] [rbp-178h]
  __int64 v51; // [rsp+F0h] [rbp-170h]
  __int64 v52; // [rsp+F8h] [rbp-168h]
  __int64 v53; // [rsp+100h] [rbp-160h]
  __int64 v54; // [rsp+108h] [rbp-158h]
  __int64 v55; // [rsp+110h] [rbp-150h]
  __int64 v56; // [rsp+118h] [rbp-148h]
  int v57; // [rsp+120h] [rbp-140h]
  __int64 v58; // [rsp+128h] [rbp-138h]
  _BYTE *v59; // [rsp+130h] [rbp-130h]
  _BYTE *v60; // [rsp+138h] [rbp-128h]
  __int64 v61; // [rsp+140h] [rbp-120h]
  int v62; // [rsp+148h] [rbp-118h]
  _BYTE v63[16]; // [rsp+150h] [rbp-110h] BYREF
  __int64 v64; // [rsp+160h] [rbp-100h]
  __int64 v65; // [rsp+168h] [rbp-F8h]
  __int64 v66; // [rsp+170h] [rbp-F0h]
  __int64 v67; // [rsp+178h] [rbp-E8h]
  __int64 v68; // [rsp+180h] [rbp-E0h]
  __int64 v69; // [rsp+188h] [rbp-D8h]
  __int16 v70; // [rsp+190h] [rbp-D0h]
  __int64 v71; // [rsp+198h] [rbp-C8h]
  __int64 v72; // [rsp+1A0h] [rbp-C0h]
  __int64 v73; // [rsp+1A8h] [rbp-B8h]
  __int64 v74; // [rsp+1B0h] [rbp-B0h]
  __int64 v75; // [rsp+1B8h] [rbp-A8h]
  int v76; // [rsp+1C0h] [rbp-A0h]
  __int64 v77; // [rsp+1C8h] [rbp-98h]
  __int64 v78; // [rsp+1D0h] [rbp-90h]
  __int64 v79; // [rsp+1D8h] [rbp-88h]
  char *v80; // [rsp+1E0h] [rbp-80h]
  __int64 v81; // [rsp+1E8h] [rbp-78h]
  char v82; // [rsp+1F0h] [rbp-70h] BYREF

  v6 = sub_13FC520(a2);
  v7 = sub_157EBA0(v6);
  memset(v35, 0, 24);
  v35[3] = sub_16498A0(v7);
  v35[4] = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  sub_17050D0(v35, v7);
  v8 = a1[2];
  v9 = *(const char ***)(v8 + 112);
  v29 = sub_1495DC0(v8, a3, a4);
  v10 = *(_QWORD *)(a1[56] + 368LL);
  v11 = sub_1456040(v29);
  v12 = sub_1643030(v11);
  v13 = v29;
  if ( v12 > (unsigned int)sub_1643030(v10) )
    v13 = sub_1483C80(v9, v29, v10, a3, a4);
  v30 = sub_14758B0((__int64)v9, v13, v10);
  v14 = sub_1456040(v30);
  v42 = sub_145CF80((__int64)v9, v14, 1, 0);
  v39 = &v41;
  v41 = (const char *)v30;
  v40 = 0x200000002LL;
  v31 = sub_147DD40((__int64)v9, (__int64 *)&v39, 0, 0, a3, a4);
  if ( v39 != &v41 )
    _libc_free((unsigned __int64)v39);
  v15 = sub_157EB90(**(_QWORD **)(a2 + 32));
  v16 = sub_1632FA0(v15);
  v39 = v9;
  v40 = v16;
  v28 = v16;
  v41 = "induction";
  v59 = v63;
  v60 = v63;
  v70 = 1;
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
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v61 = 2;
  v62 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v17 = sub_15E0530((__int64)v9[3]);
  v71 = 0;
  v74 = v17;
  v80 = &v82;
  v79 = v28;
  v73 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v72 = 0;
  v81 = 0x800000000LL;
  v18 = sub_13FC520(a2);
  v19 = sub_157EBA0(v18);
  v20 = sub_1456040((__int64)v31);
  v21 = sub_38767A0(&v39, v31, v20, v19);
  a1[54] = v21;
  v22 = v21;
  if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 15 )
  {
    v24 = sub_13FC520(a2);
    v25 = sub_157EBA0(v24);
    v26 = a1[54];
    v34 = 1;
    v32 = "exitcount.ptrcnt.to.int";
    v33 = 3;
    v27 = sub_15FDFF0(v26, v10, (__int64)&v32, v25);
    a1[54] = v27;
    v22 = v27;
  }
  sub_194A930((__int64)&v39);
  if ( v35[0] )
    sub_161E7C0((__int64)v35, v35[0]);
  return v22;
}
