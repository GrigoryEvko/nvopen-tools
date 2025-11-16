// Function: sub_2B45470
// Address: 0x2b45470
//
unsigned int *__fastcall sub_2B45470(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        char *a7,
        __int64 a8)
{
  __int64 v8; // r12
  char **v9; // r10
  unsigned int *v10; // r14
  __int64 v11; // r13
  __int64 v12; // r8
  int v13; // eax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // ebx
  __int64 v17; // r14
  __int64 v18; // r8
  unsigned __int64 v19; // xmm0_8
  unsigned __int64 v20; // r13
  unsigned __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned int *v24; // rax
  char *v25; // r12
  __int64 v26; // r8
  char *v27; // rbx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // r12
  __int64 v32; // rdx
  int v33; // ebx
  int v34; // eax
  char v35; // al
  int v36; // r9d
  int v37; // esi
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rcx
  bool v43; // al
  bool v44; // zf
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  char *v47; // rdi
  int v49; // eax
  char v50; // al
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rsi
  bool v54; // al
  unsigned __int64 v55; // [rsp+0h] [rbp-860h]
  char **v56; // [rsp+8h] [rbp-858h]
  _BYTE *v57; // [rsp+10h] [rbp-850h]
  __int64 *v58; // [rsp+18h] [rbp-848h]
  unsigned int *v59; // [rsp+20h] [rbp-840h]
  char *v60; // [rsp+28h] [rbp-838h]
  __int64 v61; // [rsp+30h] [rbp-830h]
  char *v62; // [rsp+38h] [rbp-828h]
  __int64 v63; // [rsp+40h] [rbp-820h]
  __int64 v64; // [rsp+48h] [rbp-818h]
  __int64 v65; // [rsp+50h] [rbp-810h]
  unsigned __int64 v66; // [rsp+58h] [rbp-808h]
  __int64 v67; // [rsp+60h] [rbp-800h]
  __int64 v68; // [rsp+68h] [rbp-7F8h]
  int v69; // [rsp+70h] [rbp-7F0h] BYREF
  char v70; // [rsp+74h] [rbp-7ECh]
  _BYTE *v71; // [rsp+78h] [rbp-7E8h] BYREF
  __int64 v72; // [rsp+80h] [rbp-7E0h]
  _BYTE v73[136]; // [rsp+88h] [rbp-7D8h] BYREF
  char **v74; // [rsp+110h] [rbp-750h] BYREF
  __int64 v75; // [rsp+118h] [rbp-748h]
  char *v76; // [rsp+120h] [rbp-740h] BYREF
  unsigned __int64 v77; // [rsp+128h] [rbp-738h]
  char v78; // [rsp+130h] [rbp-730h] BYREF
  char v79; // [rsp+138h] [rbp-728h] BYREF
  char *v80; // [rsp+158h] [rbp-708h]
  char v81; // [rsp+168h] [rbp-6F8h] BYREF

  v9 = &v76;
  v10 = a1;
  v11 = a2;
  v58 = a5;
  v12 = *(_QWORD *)(a2 + 80);
  v74 = &v76;
  v62 = a7;
  v64 = a3;
  v61 = a8;
  v13 = *(_DWORD *)(a3 + 32);
  v63 = a4;
  LODWORD(v60) = v13;
  v75 = 0x800000000LL;
  if ( *(_DWORD *)(v12 + 12) == 1 )
  {
    v70 = 0;
    v72 = 0x800000000LL;
    v69 = (int)v60;
    v57 = v73;
    v71 = v73;
  }
  else
  {
    v59 = a1;
    v14 = 8;
    v15 = 0;
    v57 = (_BYTE *)a2;
    v16 = 0;
    v17 = v12;
    while ( 1 )
    {
      v18 = v15 + 1;
      v19 = _mm_cvtsi32_si128(v16).m128i_u64[0];
      v20 = v8 & 0xFFFFFF0000000000LL;
      v8 &= 0xFFFFFF0000000000LL;
      if ( v15 + 1 > v14 )
      {
        v56 = v9;
        v55 = v19;
        sub_C8D5F0((__int64)&v74, v9, v15 + 1, 0x10u, v18, 0xFFFFFF00FFFFFFFFLL);
        v15 = (unsigned int)v75;
        v19 = v55;
        v9 = v56;
      }
      v21 = (unsigned __int64 *)&v74[2 * v15];
      ++v16;
      v21[1] = v20;
      *v21 = v19;
      v15 = (unsigned int)(v75 + 1);
      LODWORD(v75) = v75 + 1;
      v22 = (unsigned int)(*(_DWORD *)(v17 + 12) - 1);
      if ( (unsigned int)v22 <= v16 )
        break;
      v14 = HIDWORD(v75);
    }
    v11 = (__int64)v57;
    v70 = 0;
    v10 = v59;
    v69 = (int)v60;
    v57 = v73;
    v71 = v73;
    v72 = 0x800000000LL;
    if ( (_DWORD)v15 )
    {
      v60 = (char *)v9;
      sub_2B0D350((__int64)&v71, (__int64)&v74, v15, v22, v18, 0xFFFFFF00FFFFFFFFLL);
      v9 = (char **)v60;
    }
    if ( v74 != v9 )
    {
      v60 = (char *)v9;
      _libc_free((unsigned __int64)v74);
      v9 = (char **)v60;
    }
  }
  v59 = (unsigned int *)v9;
  v23 = sub_B43CA0(v11);
  v75 = v11;
  v74 = (char **)v23;
  v60 = &v78;
  v76 = &v78;
  v77 = 0x800000000LL;
  sub_D39570(v11, v59);
  v24 = (unsigned int *)sub_2B28460((__int64)&v74, (__int64)&v69);
  v25 = v76;
  v59 = v24;
  v26 = 224LL * (unsigned int)v77;
  v27 = &v76[v26];
  if ( v76 != &v76[v26] )
  {
    do
    {
      v27 -= 224;
      v28 = *((_QWORD *)v27 + 23);
      if ( (char *)v28 != v27 + 200 )
        j_j___libc_free_0(v28);
      v29 = *((_QWORD *)v27 + 19);
      if ( (char *)v29 != v27 + 168 )
        j_j___libc_free_0(v29);
      v30 = *((_QWORD *)v27 + 1);
      if ( (char *)v30 != v27 + 24 )
        _libc_free(v30);
    }
    while ( v25 != v27 );
    v27 = v76;
  }
  if ( v27 != v60 )
    _libc_free((unsigned __int64)v27);
  if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(v11 + 72), 23) && !(unsigned __int8)sub_B49560(v11, 23) )
  {
    if ( !v59 )
      goto LABEL_43;
LABEL_25:
    v67 = sub_DFD7B0(v63);
    v31 = v67;
    v33 = v32;
    v65 = v67;
    v68 = v32;
    v66 = (unsigned int)v32;
    v34 = sub_9B78C0(v11, v58);
    LODWORD(v60) = 0;
    LODWORD(v59) = v34;
    v35 = sub_920620(v11);
    v36 = (int)v60;
    v37 = (int)v59;
    if ( !v35 )
      goto LABEL_28;
    goto LABEL_26;
  }
  if ( ((unsigned __int8)sub_A73ED0((_QWORD *)(v11 + 72), 4) || (unsigned __int8)sub_B49560(v11, 4)) && v59 )
    goto LABEL_25;
LABEL_43:
  v31 = 0;
  v33 = 1;
  v49 = sub_9B78C0(v11, v58);
  LODWORD(v60) = 0;
  LODWORD(v59) = v49;
  v50 = sub_920620(v11);
  v36 = (int)v60;
  v37 = (int)v59;
  if ( !v50 )
    goto LABEL_44;
LABEL_26:
  v36 = *(_BYTE *)(v11 + 1) >> 1;
  if ( v36 == 127 )
    v36 = -1;
LABEL_28:
  if ( v33 )
  {
LABEL_44:
    sub_DF8CB0((__int64)&v74, v37, v64, v62, v61, v36, 0, 0x2710u);
    v51 = sub_DFD690(v63, (__int64)&v74);
    v53 = v51;
    v41 = v52;
    v42 = (unsigned int)v52;
    if ( (_DWORD)v52 )
      v54 = (int)v52 > 0;
    else
      v54 = v51 > 10000;
    v44 = !v54;
    v45 = 0;
    if ( v44 )
      v45 = v53;
    else
      v42 = 1;
    goto LABEL_35;
  }
  v65 = v31;
  v66 &= 0xFFFFFFFF00000000LL;
  sub_DF8CB0((__int64)&v74, v37, v64, v62, v61, v36, 0, __PAIR128__(v66, v31));
  v38 = sub_DFD690(v63, (__int64)&v74);
  v40 = v38;
  v41 = v39;
  v42 = (unsigned int)v39;
  if ( (_DWORD)v39 )
    v43 = (int)v39 > 0;
  else
    v43 = v38 > v31;
  v44 = !v43;
  if ( v43 )
    v42 = 1;
  v45 = 0;
  if ( v44 )
    v45 = v40;
LABEL_35:
  *(_QWORD *)v10 = v45;
  *((_QWORD *)v10 + 2) = v31;
  v46 = v41 & 0xFFFFFFFF00000000LL | v42;
  v47 = v80;
  v10[6] = v33;
  *((_QWORD *)v10 + 1) = v46;
  if ( v47 != &v81 )
    _libc_free((unsigned __int64)v47);
  if ( (char *)v77 != &v79 )
    _libc_free(v77);
  if ( v71 != v57 )
    _libc_free((unsigned __int64)v71);
  return v10;
}
