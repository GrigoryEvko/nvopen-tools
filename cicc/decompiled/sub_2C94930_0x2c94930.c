// Function: sub_2C94930
// Address: 0x2c94930
//
void __fastcall sub_2C94930(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r10
  __int64 *v10; // rbx
  __int16 v11; // ax
  __int64 *v12; // r14
  __int64 v13; // rdi
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r11
  __int16 v23; // ax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  unsigned int v28; // eax
  char v29; // cl
  char v30; // al
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  _QWORD *v42; // r14
  __int64 v43; // rax
  _QWORD *v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  _QWORD *v50; // rsi
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 *v53; // rax
  _QWORD *v54; // r14
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  _QWORD *v62; // r12
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  _QWORD *v69; // r12
  __int64 v70; // rax
  __int64 v71; // [rsp+0h] [rbp-D0h]
  _QWORD *v72; // [rsp+8h] [rbp-C8h]
  __int64 v73; // [rsp+10h] [rbp-C0h]
  __int64 *v74; // [rsp+10h] [rbp-C0h]
  __int64 v75; // [rsp+10h] [rbp-C0h]
  __int64 v76; // [rsp+10h] [rbp-C0h]
  unsigned int v77; // [rsp+18h] [rbp-B8h]
  __int64 v78; // [rsp+18h] [rbp-B8h]
  _QWORD *v79; // [rsp+18h] [rbp-B8h]
  __int64 v80; // [rsp+18h] [rbp-B8h]
  __int64 v81; // [rsp+18h] [rbp-B8h]
  __int64 v82; // [rsp+20h] [rbp-B0h]
  _QWORD *v83; // [rsp+20h] [rbp-B0h]
  __int64 v84; // [rsp+20h] [rbp-B0h]
  __int64 *v85; // [rsp+28h] [rbp-A8h]
  __int64 *v86; // [rsp+28h] [rbp-A8h]
  __int64 *v87; // [rsp+28h] [rbp-A8h]
  __int64 *v88; // [rsp+28h] [rbp-A8h]
  __int64 *v89; // [rsp+28h] [rbp-A8h]
  __int64 *v90; // [rsp+28h] [rbp-A8h]
  __int64 *v91; // [rsp+28h] [rbp-A8h]
  __int64 v92; // [rsp+28h] [rbp-A8h]
  __int64 v93; // [rsp+38h] [rbp-98h] BYREF
  __int64 v94; // [rsp+40h] [rbp-90h] BYREF
  _QWORD *v95; // [rsp+48h] [rbp-88h] BYREF
  _BYTE *v96; // [rsp+50h] [rbp-80h] BYREF
  __int64 v97; // [rsp+58h] [rbp-78h]
  _BYTE v98[112]; // [rsp+60h] [rbp-70h] BYREF

  v6 = (__int64 *)a1;
  v10 = (__int64 *)a5;
  v11 = *(_WORD *)(a1 + 24);
  if ( v11 == 5 )
  {
LABEL_2:
    v12 = (__int64 *)v6[4];
    v85 = &v12[v6[5]];
    while ( v85 != v12 )
    {
      v13 = *v12++;
      sub_2C94930(v13, a2, a3, a4, v10);
    }
    return;
  }
  while ( 1 )
  {
    if ( v11 == 8 )
    {
      v86 = v6;
      v14 = sub_D968A0(*(_QWORD *)v6[4]);
      v6 = v86;
      if ( v14 )
        goto LABEL_22;
      v82 = v86[6];
      v77 = *((_WORD *)v86 + 14) & 7;
      v17 = sub_D33D80(v86, (__int64)a4, v15, v16, v77);
      v18 = sub_D95540(*(_QWORD *)v86[4]);
      v19 = sub_DA2C50((__int64)a4, v18, 0, 0);
      v20 = sub_DC1960((__int64)a4, (__int64)v19, v17, v82, v77);
      sub_2C94930(v20, a2, a3, a4, v10);
      v6 = *(__int64 **)v86[4];
      goto LABEL_7;
    }
    if ( v11 != 6 )
      break;
    if ( v6[5] != 2 )
      goto LABEL_22;
    v21 = (__int64 *)v6[4];
    if ( *(_WORD *)(*v21 + 24) )
      goto LABEL_22;
    v87 = v6;
    if ( a2 )
    {
      a2 = sub_DCA690(a4, (__int64)a2, *v21, 0, 0);
      v21 = (__int64 *)v87[4];
    }
    else
    {
      a2 = (__int64 *)*v21;
    }
    v6 = (__int64 *)v21[1];
LABEL_7:
    v11 = *((_WORD *)v6 + 12);
    if ( v11 == 5 )
      goto LABEL_2;
  }
  if ( v11 == 4 )
  {
    v22 = v6[4];
    v23 = *(_WORD *)(v22 + 24);
    if ( ((unsigned __int16)(v23 - 5) <= 1u || (unsigned __int16)(v23 - 8) <= 5u) && (*(_BYTE *)(v22 + 28) & 4) == 0 )
    {
      v92 = (__int64)v6;
      v84 = v6[4];
      v56 = sub_D95540(v84);
      v57 = sub_D97050((__int64)a4, v56);
      v6 = (__int64 *)v92;
      if ( v57 <= 0x1F || (v22 = v84, !(_BYTE)qword_5012128) )
      {
        if ( a2 )
          v6 = sub_DCA690(a4, (__int64)a2, v92, 0, 0);
        sub_D9B3A0(a3, (__int64)v6, v58, v59, v60, v61);
        return;
      }
    }
    v73 = (__int64)v6;
    v96 = v98;
    v78 = v22;
    v97 = 0x800000000LL;
    v24 = sub_D95540(v22);
    v95 = sub_DA2C50((__int64)a4, v24, 0, 0);
    v83 = v95;
    sub_2C94930(v78, 0, &v96, a4, &v95);
    v6 = (__int64 *)v73;
    if ( v83 == v95 )
    {
LABEL_20:
      if ( v96 != v98 )
      {
        v88 = v6;
        _libc_free((unsigned __int64)v96);
        v6 = v88;
      }
      goto LABEL_22;
    }
    v51 = sub_D95540(v73);
    v52 = sub_DC5000((__int64)a4, (__int64)v95, v51, 0);
    if ( a2 )
    {
      v53 = sub_DCA690(a4, (__int64)a2, (__int64)v52, 0, 0);
      *v10 = (__int64)sub_DC7ED0(a4, *v10, (__int64)v53, 0, 0);
      v54 = sub_DC7EB0(a4, (__int64)&v96, 0, 0);
      v55 = sub_D95540(v73);
      v44 = sub_DC5000((__int64)a4, (__int64)v54, v55, 0);
      goto LABEL_43;
    }
    *v10 = (__int64)sub_DC7ED0(a4, *v10, (__int64)v52, 0, 0);
    v62 = sub_DC7EB0(a4, (__int64)&v96, 0, 0);
    v63 = sub_D95540(v73);
    v50 = sub_DC5000((__int64)a4, (__int64)v62, v63, 0);
    goto LABEL_45;
  }
  if ( v11 )
  {
    if ( (_BYTE)qword_50129E8 != 1 )
      goto LABEL_22;
    if ( v11 != 3 )
      goto LABEL_22;
    v26 = v6[4];
    v89 = v6;
    v94 = 0;
    v27 = sub_D95540(v26);
    v28 = sub_D97050((__int64)a4, v27);
    v6 = v89;
    v29 = v28;
    if ( v28 > 0x20 )
      goto LABEL_22;
    if ( !byte_5012208 || v28 != 32 || (v64 = sub_D97050((__int64)a4, v89[5]), v6 = v89, v29 = 32, v64 != 64) )
    {
      v90 = v6;
      v30 = sub_2C922A0(v26, (__int64)a4, &v93, (1LL << v29) - 1, (unsigned __int64 *)&v94);
      v6 = v90;
      if ( !v30 )
      {
LABEL_22:
        if ( a2 )
          v6 = sub_DCA690(a4, (__int64)a2, (__int64)v6, 0, 0);
        v25 = *(unsigned int *)(a3 + 8);
        if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v91 = v6;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v25 + 1, 8u, a5, a6);
          v25 = *(unsigned int *)(a3 + 8);
          v6 = v91;
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v25) = v6;
        ++*(_DWORD *)(a3 + 8);
        return;
      }
    }
    v74 = v6;
    v96 = v98;
    v97 = 0x800000000LL;
    v31 = sub_D95540(v26);
    v79 = sub_DA2C50((__int64)a4, v31, 0, 0);
    v95 = v79;
    sub_2C94930(v26, 0, &v96, a4, &v95);
    v6 = v74;
    if ( v79 == v95 )
      goto LABEL_20;
    a5 = sub_2C90D50(*(_QWORD *)(v95[4] + 24LL), *(_DWORD *)(v95[4] + 32LL));
    if ( a5 <= (int)v94 )
      goto LABEL_20;
    v71 = (__int64)v6;
    v75 = v94;
    v32 = sub_D95540(v26);
    v72 = sub_DA2C50((__int64)a4, v32, v75, 0);
    v33 = sub_2C90D50(*(_QWORD *)(v95[4] + 24LL), *(_DWORD *)(v95[4] + 32LL));
    v76 = v33 - v94;
    v34 = sub_D95540(v26);
    v95 = sub_DA2C50((__int64)a4, v34, v76, 0);
    v35 = sub_D95540(v71);
    v36 = sub_DC2B70((__int64)a4, (__int64)v95, v35, 0);
    if ( a2 )
    {
      v37 = sub_DCA690(a4, (__int64)a2, (__int64)v36, 0, 0);
      *v10 = (__int64)sub_DC7ED0(a4, *v10, (__int64)v37, 0, 0);
      if ( v79 == v72 )
      {
        v80 = v71;
      }
      else
      {
        v80 = v71;
        sub_D9B3A0((__int64)&v96, (__int64)v72, v38, v39, v40, v41);
      }
      v42 = sub_DC7EB0(a4, (__int64)&v96, 0, 0);
      v43 = sub_D95540(v80);
      v44 = sub_DC2B70((__int64)a4, (__int64)v42, v43, 0);
LABEL_43:
      v45 = sub_DCA690(a4, (__int64)a2, (__int64)v44, 0, 0);
    }
    else
    {
      *v10 = (__int64)sub_DC7ED0(a4, *v10, (__int64)v36, 0, 0);
      if ( v79 == v72 )
      {
        v81 = v71;
      }
      else
      {
        v81 = v71;
        sub_D9B3A0((__int64)&v96, (__int64)v72, v65, v66, v67, v68);
      }
      v69 = sub_DC7EB0(a4, (__int64)&v96, 0, 0);
      v70 = sub_D95540(v81);
      v45 = sub_DC2B70((__int64)a4, (__int64)v69, v70, 0);
    }
    v50 = v45;
LABEL_45:
    sub_D9B3A0(a3, (__int64)v50, v46, v47, v48, v49);
    if ( v96 != v98 )
      _libc_free((unsigned __int64)v96);
  }
  else
  {
    if ( a2 )
      v6 = sub_DCA690(a4, (__int64)a2, (__int64)v6, 0, 0);
    *v10 = (__int64)sub_DC7ED0(a4, *v10, (__int64)v6, 0, 0);
  }
}
