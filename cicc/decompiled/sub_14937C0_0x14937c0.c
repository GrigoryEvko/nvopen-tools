// Function: sub_14937C0
// Address: 0x14937c0
//
__int64 __fastcall sub_14937C0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        __m128i a8,
        char a9,
        char a10)
{
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  bool v18; // al
  __int64 v19; // r9
  __int64 *v20; // rax
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rax
  unsigned int v29; // r13d
  int v30; // eax
  int v31; // eax
  __int64 *v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  bool v36; // al
  __int64 v37; // rdx
  __int64 v38; // rax
  char v39; // al
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  unsigned int v46; // r13d
  int v47; // eax
  int v48; // eax
  __int64 *v49; // rax
  char v50; // al
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-110h]
  __int64 v53; // [rsp+0h] [rbp-110h]
  char v54; // [rsp+8h] [rbp-108h]
  __int64 v56; // [rsp+10h] [rbp-100h]
  __int64 v57; // [rsp+10h] [rbp-100h]
  __int64 v58; // [rsp+20h] [rbp-F0h]
  __int64 v59; // [rsp+20h] [rbp-F0h]
  __int64 v60; // [rsp+20h] [rbp-F0h]
  __int64 v61; // [rsp+20h] [rbp-F0h]
  __int64 v63; // [rsp+28h] [rbp-E8h]
  __int64 v64[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v65[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+50h] [rbp-C0h] BYREF
  int v67; // [rsp+58h] [rbp-B8h]
  __int64 v68[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v69; // [rsp+70h] [rbp-A0h] BYREF
  int v70; // [rsp+78h] [rbp-98h]
  __int64 v71; // [rsp+80h] [rbp-90h] BYREF
  int v72; // [rsp+88h] [rbp-88h]
  __int64 v73; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v74; // [rsp+98h] [rbp-78h]
  _BYTE *v75; // [rsp+A0h] [rbp-70h]
  __int64 v76; // [rsp+A8h] [rbp-68h]
  int v77; // [rsp+B0h] [rbp-60h]
  _BYTE v78[88]; // [rsp+B8h] [rbp-58h] BYREF

  v54 = a9;
  v73 = 0;
  v74 = v78;
  v75 = v78;
  v76 = 4;
  v77 = 0;
  if ( !sub_146CEE0((__int64)a2, a4, a5) )
    goto LABEL_4;
  v12 = a3;
  if ( *(_WORD *)(a3 + 24) == 7 )
  {
    if ( a5 != *(_QWORD *)(a3 + 48) )
    {
LABEL_4:
      v13 = sub_1456E90((__int64)a2);
      sub_14573F0(a1, v13);
      goto LABEL_5;
    }
  }
  else
  {
    if ( !a10 )
      goto LABEL_4;
    v15 = sub_1493080((__int64)a2, a3, a5, (__int64)&v73, a7, a8);
    v12 = (__int64)v15;
    if ( !v15 || a5 != v15[6] )
      goto LABEL_4;
  }
  if ( *(_QWORD *)(v12 + 40) != 2 )
    goto LABEL_4;
  if ( a9 )
    v54 = ((a6 == 0 ? 2 : 4) & *(unsigned __int16 *)(v12 + 26)) != 0;
  v58 = v12;
  v16 = sub_13A5BC0((_QWORD *)v12, (__int64)a2);
  v17 = sub_1480620((__int64)a2, v16, 0);
  if ( !(unsigned __int8)sub_1477C30((__int64)a2, v17) )
    goto LABEL_4;
  v18 = sub_1456110(v17);
  v19 = v58;
  if ( !v18 )
  {
    v50 = sub_14822A0((__int64)a2, a4, v17, a6, v54);
    v19 = v58;
    if ( v50 )
      goto LABEL_4;
  }
  v20 = *(__int64 **)(v19 + 32);
  if ( !a6 )
  {
    v61 = *v20;
    v38 = sub_13A5B00((__int64)a2, *v20, v17, 0, 0);
    v39 = sub_148B410((__int64)a2, a5, 0x22u, v38, a4);
    v40 = v61;
    if ( v39 )
    {
      v42 = a4;
    }
    else
    {
      v41 = sub_1481BD0(a2, a4, v61, a7, a8);
      v40 = v61;
      v42 = v41;
    }
    v57 = v40;
    v43 = sub_14806B0((__int64)a2, v40, v42, 0, 0);
    v60 = sub_1484BE0(a2, v43, v17, 0, a7, a8);
    sub_1477A60((__int64)v64, (__int64)a2, v57);
    v44 = sub_1477920((__int64)a2, v17, 0);
    sub_158AAD0(v65, v44);
    v45 = sub_1456040(a3);
    v46 = sub_1456C90((__int64)a2, v45);
    sub_13A38D0((__int64)&v69, (__int64)v65);
    sub_16A7800(&v69, 1);
    v47 = v70;
    v70 = 0;
    v72 = v47;
    v71 = v69;
    sub_135E0D0((__int64)v68, v46, 0, 0);
    sub_16A7200(&v71, v68);
    v48 = v72;
    v72 = 0;
    v67 = v48;
    v66 = v71;
    sub_135E100(v68);
    sub_135E100(&v71);
    sub_135E100(&v69);
    v49 = sub_1477920((__int64)a2, a4, 0);
    sub_158AAD0(&v71, v49);
    if ( (int)sub_16A9900(&v71, &v66) > 0 )
      goto LABEL_20;
LABEL_29:
    sub_13A38D0((__int64)v68, (__int64)&v66);
    goto LABEL_21;
  }
  v59 = *v20;
  v21 = sub_13A5B00((__int64)a2, *v20, v17, 0, 0);
  v22 = sub_148B410((__int64)a2, a5, 0x26u, v21, a4);
  v23 = v59;
  if ( v22 )
  {
    v24 = a4;
  }
  else
  {
    v51 = sub_1480950(a2, a4, v59, a7, a8);
    v23 = v59;
    v24 = v51;
  }
  v56 = v23;
  v25 = sub_14806B0((__int64)a2, v23, v24, 0, 0);
  v60 = sub_1484BE0(a2, v25, v17, 0, a7, a8);
  v26 = sub_1477920((__int64)a2, v56, 1u);
  sub_158ABC0(v64, v26);
  v27 = sub_1477920((__int64)a2, v17, 1u);
  sub_158ACE0(v65, v27);
  v28 = sub_1456040(a3);
  v29 = sub_1456C90((__int64)a2, v28);
  sub_13A38D0((__int64)&v69, (__int64)v65);
  sub_16A7800(&v69, 1);
  v30 = v70;
  v70 = 0;
  v72 = v30;
  v71 = v69;
  sub_13D00B0((__int64)v68, v29);
  sub_16A7200(&v71, v68);
  v31 = v72;
  v72 = 0;
  v67 = v31;
  v66 = v71;
  sub_135E100(v68);
  sub_135E100(&v71);
  sub_135E100(&v69);
  v32 = sub_1477920((__int64)a2, a4, 1u);
  sub_158ACE0(&v71, v32);
  if ( (int)sub_16AEA10(&v71, &v66) <= 0 )
    goto LABEL_29;
LABEL_20:
  sub_13A38D0((__int64)v68, (__int64)&v71);
LABEL_21:
  sub_135E100(&v71);
  sub_1456E90((__int64)a2);
  v33 = v60;
  if ( *(_WORD *)(v60 + 24) )
  {
    v52 = sub_145CF40((__int64)a2, (__int64)v65);
    sub_13A38D0((__int64)&v69, (__int64)v64);
    sub_16A7590(&v69, v68);
    v34 = v70;
    v70 = 0;
    v72 = v34;
    v71 = v69;
    v35 = sub_145CF40((__int64)a2, (__int64)&v71);
    v53 = sub_1484BE0(a2, v35, v52, 0, a7, a8);
    sub_135E100(&v71);
    sub_135E100(&v69);
    v33 = v53;
  }
  v63 = v33;
  v36 = sub_14562D0(v33);
  v37 = v63;
  if ( v36 )
    v37 = v60;
  sub_14575B0(a1, v60, v37, 0, (__int64)&v73);
  sub_135E100(v68);
  sub_135E100(&v66);
  sub_135E100(v65);
  sub_135E100(v64);
LABEL_5:
  if ( v75 != v74 )
    _libc_free((unsigned __int64)v75);
  return a1;
}
