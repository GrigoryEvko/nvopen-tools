// Function: sub_12B1C20
// Address: 0x12b1c20
//
__int64 __fastcall sub_12B1C20(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  char v9; // bl
  char *v10; // r15
  char *v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r14
  __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // r15
  __int64 v29; // rdi
  unsigned __int64 *v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // r14
  __int64 v38; // rdi
  unsigned __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rsi
  _QWORD *v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rax
  _BYTE *v47; // rsi
  _BYTE *v48; // rdi
  char *v50; // [rsp+10h] [rbp-1D0h]
  char *v51; // [rsp+18h] [rbp-1C8h]
  __int64 v53; // [rsp+20h] [rbp-1C0h]
  __int64 v54; // [rsp+20h] [rbp-1C0h]
  unsigned __int64 *v55; // [rsp+20h] [rbp-1C0h]
  char *v56; // [rsp+28h] [rbp-1B8h]
  __int64 v57; // [rsp+30h] [rbp-1B0h]
  __int64 v58; // [rsp+38h] [rbp-1A8h]
  __int64 v59; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v60; // [rsp+48h] [rbp-198h] BYREF
  __int64 v61; // [rsp+50h] [rbp-190h] BYREF
  unsigned __int64 v62; // [rsp+58h] [rbp-188h] BYREF
  _QWORD v63[4]; // [rsp+60h] [rbp-180h] BYREF
  _BYTE v64[16]; // [rsp+80h] [rbp-160h] BYREF
  __int16 v65; // [rsp+90h] [rbp-150h]
  _BYTE *v66; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-138h]
  _BYTE v68[304]; // [rsp+B0h] [rbp-130h] BYREF

  v6 = *(_QWORD *)(a4 + 72);
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 16LL);
  v57 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(v7 + 16);
  v58 = *(_QWORD *)(v8 + 16);
  v9 = sub_12A6F10(
         *(_QWORD *)(v58 + 16),
         3u,
         "unexpected 'rowcol' operand",
         "'rowcol' operand can be 0, 1, 2, or 3 only",
         (_DWORD *)(a4 + 36));
  v56 = sub_128F980((__int64)a2, v57);
  v10 = sub_128F980((__int64)a2, v7);
  v51 = sub_128F980((__int64)a2, v8);
  v11 = sub_128F980((__int64)a2, v58);
  v12 = a2[4];
  v50 = v11;
  v66 = v68;
  v67 = 0x2000000000LL;
  sub_12B1610(v12, a3, v9, &v62, &v59, &v60, &v61);
  v13 = v62;
  v14 = sub_1643360(a2[5]);
  v15 = sub_159C470(v14, v13, 0);
  v16 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    sub_16CD150(&v66, v68, 0, 8);
    v16 = (unsigned int)v67;
  }
  *(_QWORD *)&v66[8 * v16] = v15;
  v17 = v59;
  LODWORD(v67) = v67 + 1;
  if ( *(_BYTE *)(v59 + 8) == 16 )
  {
    v19 = (_QWORD *)sub_12A8C20(a2, v59, v10);
    v26 = (unsigned int)v67;
    if ( (unsigned int)v67 < HIDWORD(v67) )
      goto LABEL_14;
    goto LABEL_40;
  }
  v65 = 257;
  v18 = sub_1648A60(64, 1);
  v19 = (_QWORD *)v18;
  if ( v18 )
    sub_15F9210(v18, v17, v10, 0, 0, 0);
  v20 = a2[7];
  if ( v20 )
  {
    v21 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v20 + 40, v19);
    v22 = v19[3];
    v23 = *v21;
    v19[4] = v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    v19[3] = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v19 + 3;
    *v21 = *v21 & 7 | (unsigned __int64)(v19 + 3);
  }
  sub_164B780(v19, v64);
  v24 = a2[6];
  if ( v24 )
  {
    v63[0] = a2[6];
    sub_1623A60(v63, v24, 2);
    if ( v19[6] )
      sub_161E7C0(v19 + 6);
    v25 = v63[0];
    v19[6] = v63[0];
    if ( v25 )
      sub_1623210(v63, v25, v19 + 6);
  }
  v26 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
LABEL_40:
    sub_16CD150(&v66, v68, 0, 8);
    v26 = (unsigned int)v67;
  }
LABEL_14:
  *(_QWORD *)&v66[8 * v26] = v19;
  LODWORD(v67) = v67 + 1;
  if ( *(_BYTE *)(v60 + 8) == 16 )
  {
    v28 = (_QWORD *)sub_12A8C20(a2, v60, v51);
    v35 = (unsigned int)v67;
    if ( (unsigned int)v67 < HIDWORD(v67) )
      goto LABEL_25;
    goto LABEL_42;
  }
  v53 = v60;
  v65 = 257;
  v27 = sub_1648A60(64, 1);
  v28 = (_QWORD *)v27;
  if ( v27 )
    sub_15F9210(v27, v53, v51, 0, 0, 0);
  v29 = a2[7];
  if ( v29 )
  {
    v30 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v29 + 40, v28);
    v31 = v28[3];
    v32 = *v30;
    v28[4] = v30;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    v28[3] = v32 | v31 & 7;
    *(_QWORD *)(v32 + 8) = v28 + 3;
    *v30 = *v30 & 7 | (unsigned __int64)(v28 + 3);
  }
  sub_164B780(v28, v64);
  v33 = a2[6];
  if ( v33 )
  {
    v63[0] = a2[6];
    sub_1623A60(v63, v33, 2);
    if ( v28[6] )
      sub_161E7C0(v28 + 6);
    v34 = v63[0];
    v28[6] = v63[0];
    if ( v34 )
      sub_1623210(v63, v34, v28 + 6);
  }
  v35 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
LABEL_42:
    sub_16CD150(&v66, v68, 0, 8);
    v35 = (unsigned int)v67;
  }
LABEL_25:
  *(_QWORD *)&v66[8 * v35] = v28;
  LODWORD(v67) = v67 + 1;
  if ( *(_BYTE *)(v61 + 8) == 16 )
  {
    v37 = (_QWORD *)sub_12A8C20(a2, v61, v50);
    v44 = (unsigned int)v67;
    if ( (unsigned int)v67 < HIDWORD(v67) )
      goto LABEL_36;
    goto LABEL_44;
  }
  v54 = v61;
  v65 = 257;
  v36 = sub_1648A60(64, 1);
  v37 = (_QWORD *)v36;
  if ( v36 )
    sub_15F9210(v36, v54, v50, 0, 0, 0);
  v38 = a2[7];
  if ( v38 )
  {
    v55 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v38 + 40, v37);
    v39 = *v55;
    v40 = v37[3] & 7LL;
    v37[4] = v55;
    v39 &= 0xFFFFFFFFFFFFFFF8LL;
    v37[3] = v39 | v40;
    *(_QWORD *)(v39 + 8) = v37 + 3;
    *v55 = *v55 & 7 | (unsigned __int64)(v37 + 3);
  }
  sub_164B780(v37, v64);
  v41 = a2[6];
  if ( v41 )
  {
    v63[0] = a2[6];
    sub_1623A60(v63, v41, 2);
    v42 = v37 + 6;
    if ( v37[6] )
    {
      sub_161E7C0(v37 + 6);
      v42 = v37 + 6;
    }
    v43 = v63[0];
    v37[6] = v63[0];
    if ( v43 )
      sub_1623210(v63, v43, v42);
  }
  v44 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
LABEL_44:
    sub_16CD150(&v66, v68, 0, 8);
    v44 = (unsigned int)v67;
  }
LABEL_36:
  *(_QWORD *)&v66[8 * v44] = v37;
  v45 = (_QWORD *)a2[4];
  v63[0] = v61;
  LODWORD(v67) = v67 + 1;
  v63[1] = v59;
  v63[2] = v60;
  v46 = sub_126A190(v45, 4173, (__int64)v63, 3u);
  v65 = 257;
  v47 = (_BYTE *)sub_1285290(a2 + 6, *(_QWORD *)(v46 + 24), v46, (int)v66, (unsigned int)v67, (__int64)v64, 0);
  sub_12A9060(a2, v47, v56);
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 12) &= ~1u;
  v48 = v66;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v48 != v68 )
    _libc_free(v48, v47);
  return a1;
}
