// Function: sub_2C20440
// Address: 0x2c20440
//
__int64 *__fastcall sub_2C20440(__int64 a1, __int64 a2)
{
  int v3; // eax
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // r12
  int v8; // r13d
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // cc
  _QWORD *v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r12
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // r14
  __int64 v24; // rax
  unsigned int **v25; // r13
  __int64 v26; // rbx
  _BYTE *v27; // rax
  __int64 v28; // rax
  unsigned int **v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rax
  unsigned int **v32; // r14
  _BYTE *v33; // r13
  _BYTE *v34; // rax
  __int64 v35; // rax
  unsigned int **v36; // r13
  __int64 v37; // rsi
  _BYTE *v38; // r14
  _BYTE *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int **v45; // r13
  __int64 v46; // rsi
  _BYTE *v47; // rax
  _BYTE *v48; // rax
  unsigned int **v49; // rdi
  __int64 v50; // r13
  _QWORD *v51; // rbx
  unsigned __int16 v52; // r9
  __int64 v53; // r10
  __int64 v54; // rdx
  int v55; // eax
  char v56; // al
  int v57; // edx
  __int64 v58; // rax
  unsigned __int16 v59; // [rsp+0h] [rbp-120h]
  __int64 v60; // [rsp+8h] [rbp-118h]
  __int64 v61; // [rsp+10h] [rbp-110h]
  __int64 v62; // [rsp+18h] [rbp-108h]
  __int64 v63; // [rsp+20h] [rbp-100h]
  _BYTE *v64; // [rsp+28h] [rbp-F8h]
  _BYTE *v65; // [rsp+30h] [rbp-F0h]
  __int64 v66; // [rsp+40h] [rbp-E0h]
  __int64 v67; // [rsp+48h] [rbp-D8h]
  _BYTE *v68; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v69; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v70; // [rsp+80h] [rbp-A0h]
  _BYTE v71[32]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v72; // [rsp+B0h] [rbp-70h]
  const char *v73[4]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v74; // [rsp+E0h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 56);
  if ( v3 != 4 )
  {
    if ( !v3 )
      BUG();
    v4 = *(_QWORD **)(a1 + 48);
    goto LABEL_13;
  }
  v4 = *(_QWORD **)(a1 + 48);
  v5 = v4[3];
  if ( !v5 )
  {
LABEL_13:
    v9 = *(_QWORD *)(*v4 + 40LL);
    v66 = *(_QWORD *)(v9 + 8);
    v61 = sub_2BF3650(a2 + 96, a1);
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(v5 + 40);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = (int)v7;
  v9 = *(_QWORD *)(*v4 + 40LL);
  v66 = *(_QWORD *)(v9 + 8);
  v61 = sub_2BF3650(a2 + 96, a1);
  if ( (_DWORD)v7 )
  {
    if ( *(_DWORD *)(a1 + 56) != 4 )
      goto LABEL_26;
    v10 = *(_QWORD *)(a1 + 48);
    v11 = *(_QWORD *)(v10 + 24);
    if ( !v11 )
      goto LABEL_26;
    v12 = *(_QWORD *)(v11 + 40);
    v13 = *(_DWORD *)(v12 + 32) <= 0x40u;
    v14 = *(_QWORD **)(v12 + 24);
    if ( !v13 )
      v14 = (_QWORD *)*v14;
    if ( (_DWORD)v14 )
    {
      v15 = *(_QWORD *)(v10 + 16);
      v62 = a1 + 96;
    }
    else
    {
LABEL_26:
      v15 = a1 + 96;
      v62 = a1 + 96;
    }
    v43 = sub_2BFB640(a2, v15, 0);
    v67 = *(_QWORD *)(v43 - 32LL * (*(_DWORD *)(v43 + 4) & 0x7FFFFFF));
    v63 = (unsigned int)v7;
    goto LABEL_24;
  }
LABEL_14:
  v16 = (_QWORD *)sub_2BF9BD0(*(_QWORD *)(a1 + 80));
  v17 = sub_2BF3F10(v16);
  v18 = *(_QWORD *)(sub_2BF04D0(v17) + 120);
  if ( v18 )
    v18 += 72;
  v19 = sub_2BFB640(a2, v18, 1) + 24;
  v74 = 259;
  v73[0] = "pointer.phi";
  v67 = sub_BD2DA0(80);
  if ( v67 )
  {
    sub_B44260(v67, v66, 55, 0x8000000u, v19, 0);
    *(_DWORD *)(v67 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v67, v73);
    sub_BD2A10(v67, *(_DWORD *)(v67 + 72), 1);
  }
  sub_F0A850(v67, v9, v61);
  v73[0] = *(const char **)(a1 + 88);
  if ( v73[0] )
    sub_2AAAFA0((__int64 *)v73);
  if ( (const char **)(v67 + 48) != v73 )
  {
    sub_9C6650((_QWORD *)(v67 + 48));
    v20 = (unsigned __int8 *)v73[0];
    *(const char **)(v67 + 48) = v73[0];
    if ( v20 )
    {
      sub_B976B0((__int64)v73, v20, v67 + 48);
      v73[0] = 0;
    }
  }
  v8 = 0;
  sub_9C6650(v73);
  v63 = 0;
  v62 = a1 + 96;
LABEL_24:
  v21 = *(_QWORD *)(a2 + 904);
  v22 = *(_QWORD *)(v21 + 56);
  LOWORD(v21) = *(_WORD *)(v21 + 64);
  LODWORD(v73[0]) = 0;
  BYTE4(v73[0]) = 0;
  v59 = v21;
  v60 = v22;
  v65 = (_BYTE *)sub_2BFB120(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), (unsigned int *)v73);
  v23 = (__int64 *)sub_2BFD6A0(a2 + 976, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL));
  v64 = (_BYTE *)sub_2AB2710(*(_QWORD *)(a2 + 904), (__int64)v23, *(_QWORD *)(a2 + 8));
  if ( !v8 )
  {
    v44 = sub_2BF9BD0(*(_QWORD *)(a1 + 80));
    v45 = *(unsigned int ***)(a2 + 904);
    v46 = **(unsigned int **)(v44 + 144);
    v74 = 257;
    v47 = (_BYTE *)sub_AD64C0((__int64)v23, v46, 0);
    v48 = (_BYTE *)sub_A81850(v45, v64, v47, (__int64)v73, 0, 0);
    v72 = 257;
    v49 = *(unsigned int ***)(a2 + 904);
    v74 = 259;
    v73[0] = "ptr.ind";
    v68 = (_BYTE *)sub_A81850(v49, v65, v48, (__int64)v71, 0, 0);
    v50 = sub_BCB2B0(*(_QWORD **)(*(_QWORD *)(a2 + 904) + 72LL));
    v51 = sub_BD2C40(88, 2u);
    if ( !v51 )
    {
LABEL_33:
      sub_F0A850(v67, (__int64)v51, v61);
      goto LABEL_25;
    }
    v52 = v59;
    v53 = *(_QWORD *)(v67 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v53 + 8) - 17 <= 1 )
    {
LABEL_32:
      sub_B44260((__int64)v51, v53, 34, 2u, v60, v52);
      v51[9] = v50;
      v51[10] = sub_B4DC50(v50, (__int64)&v68, 1);
      sub_B4D9A0((__int64)v51, v67, (__int64 *)&v68, 1, (__int64)v73);
      goto LABEL_33;
    }
    v54 = *((_QWORD *)v68 + 1);
    v55 = *(unsigned __int8 *)(v54 + 8);
    if ( v55 == 17 )
    {
      v56 = 0;
    }
    else
    {
      if ( v55 != 18 )
        goto LABEL_32;
      v56 = 1;
    }
    v57 = *(_DWORD *)(v54 + 32);
    BYTE4(v69) = v56;
    LODWORD(v69) = v57;
    v58 = sub_BCE1B0((__int64 *)v53, v69);
    v52 = v59;
    v53 = v58;
    goto LABEL_32;
  }
LABEL_25:
  v24 = sub_BCE1B0(v23, *(_QWORD *)(a2 + 8));
  v25 = *(unsigned int ***)(a2 + 904);
  v26 = v24;
  v74 = 257;
  v27 = (_BYTE *)sub_AD64C0((__int64)v23, v63, 0);
  v28 = sub_A81850(v25, v64, v27, (__int64)v73, 0, 0);
  v29 = *(unsigned int ***)(a2 + 904);
  v30 = *(_QWORD *)(a2 + 8);
  v74 = 257;
  v31 = sub_B37620(v29, v30, v28, (__int64 *)v73);
  v32 = *(unsigned int ***)(a2 + 904);
  v74 = 257;
  v33 = (_BYTE *)v31;
  v72 = 257;
  v34 = (_BYTE *)sub_B33FB0((__int64)v32, v26, (__int64)v71);
  v35 = sub_929C50(v32, v33, v34, (__int64)v73, 0, 0);
  v36 = *(unsigned int ***)(a2 + 904);
  v37 = *(_QWORD *)(a2 + 8);
  v38 = (_BYTE *)v35;
  v72 = 257;
  v70 = 257;
  v73[0] = "vector.gep";
  v74 = 259;
  v39 = (_BYTE *)sub_B37620(v36, v37, (__int64)v65, &v69);
  v68 = (_BYTE *)sub_A81850(v36, v38, v39, (__int64)v71, 0, 0);
  v40 = sub_BCB2B0(*(_QWORD **)(*(_QWORD *)(a2 + 904) + 72LL));
  v41 = sub_921130(v36, v40, v67, &v68, 1, (__int64)v73, 0);
  return sub_2BF26E0(a2, v62, v41, 0);
}
