// Function: sub_2292280
// Address: 0x2292280
//
__int64 __fastcall sub_2292280(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, _BYTE *a5)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rbx
  bool v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 *v16; // r13
  __int64 v17; // r15
  _QWORD *v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // r8
  _QWORD *v23; // rax
  bool v24; // al
  unsigned int v25; // r9d
  char v26; // al
  __int64 v27; // rcx
  __int64 v28; // r8
  bool v30; // al
  __int64 v31; // rcx
  __int64 v32; // r8
  _QWORD *v33; // rax
  __int64 *v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  _QWORD *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // r8
  _QWORD *v41; // rax
  __int64 *v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  _QWORD *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r8
  _QWORD *v49; // rax
  __int64 *v50; // r13
  __int64 *v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rcx
  __int64 *v54; // rax
  __int64 v55; // r8
  _QWORD *v56; // rax
  __int64 v57; // rcx
  __int64 v58; // r8
  _QWORD *v59; // rax
  unsigned int v60; // eax
  unsigned __int8 v61; // [rsp+7h] [rbp-89h]
  _QWORD *v62; // [rsp+8h] [rbp-88h]
  _QWORD *v63; // [rsp+8h] [rbp-88h]
  bool v64; // [rsp+10h] [rbp-80h]
  __int64 *v65; // [rsp+10h] [rbp-80h]
  bool v66; // [rsp+10h] [rbp-80h]
  bool v67; // [rsp+10h] [rbp-80h]
  __int64 *v68; // [rsp+10h] [rbp-80h]
  _QWORD *v69; // [rsp+10h] [rbp-80h]
  unsigned __int8 v73; // [rsp+28h] [rbp-68h]
  unsigned __int8 v74; // [rsp+28h] [rbp-68h]
  unsigned __int64 v75; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v76; // [rsp+38h] [rbp-58h]
  unsigned __int64 v77; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v78; // [rsp+48h] [rbp-48h]
  unsigned __int64 v79; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v80; // [rsp+58h] [rbp-38h]

  v6 = sub_228CE20(a4);
  v7 = sub_228CDE0(a4);
  v8 = sub_228CDF0(a4);
  v9 = sub_228CE00(a4);
  v10 = sub_D968A0(v7);
  if ( v10 )
  {
    if ( !*(_WORD *)(v8 + 24) && !*(_WORD *)(v9 + 24) )
    {
      v11 = *(_QWORD *)(v8 + 32);
      v76 = *(_DWORD *)(v11 + 32);
      if ( v76 > 0x40 )
      {
        v67 = v10;
        sub_C43780((__int64)&v75, (const void **)(v11 + 24));
        v10 = v67;
      }
      else
      {
        v75 = *(_QWORD *)(v11 + 24);
      }
      v12 = *(_QWORD *)(v9 + 32);
      v78 = *(_DWORD *)(v12 + 32);
      if ( v78 > 0x40 )
      {
        v66 = v10;
        sub_C43780((__int64)&v77, (const void **)(v12 + 24));
        v10 = v66;
      }
      else
      {
        v77 = *(_QWORD *)(v12 + 24);
      }
      v64 = v10;
      sub_C4A3E0((__int64)&v79, (__int64)&v77, (__int64)&v75);
      v15 = sub_2291EA0(a1, *a3, v6, v13, v14);
      v16 = *(__int64 **)(a1 + 8);
      v17 = (__int64)v15;
      v18 = sub_DA26C0(v16, (__int64)&v79);
      v19 = sub_DCA690(v16, v17, (__int64)v18, 0, 0);
      *a2 = (__int64)sub_DCC810(v16, *a2, (__int64)v19, 0, 0);
      *a3 = (__int64)sub_2291F00(a1, *a3, v6, v20);
      v23 = sub_2291EA0(a1, *a2, v6, v21, v22);
      v24 = sub_D968A0((__int64)v23);
      v25 = v64;
      if ( !v24 )
        *a5 = 0;
      if ( v80 > 0x40 && v79 )
      {
        j_j___libc_free_0_0(v79);
        v25 = v64;
      }
      if ( v78 > 0x40 && v77 )
      {
        v73 = v25;
        j_j___libc_free_0_0(v77);
        v25 = v73;
      }
      if ( v76 > 0x40 && v75 )
      {
        v74 = v25;
        j_j___libc_free_0_0(v75);
        return v74;
      }
      return v25;
    }
    return 0;
  }
  v30 = sub_D968A0(v8);
  if ( v30 )
  {
    v25 = 0;
    if ( *(_WORD *)(v7 + 24) || *(_WORD *)(v9 + 24) )
      return v25;
    v61 = v30;
    sub_9865C0((__int64)&v75, *(_QWORD *)(v7 + 32) + 24LL);
    sub_9865C0((__int64)&v77, *(_QWORD *)(v9 + 32) + 24LL);
    sub_C4A3E0((__int64)&v79, (__int64)&v77, (__int64)&v75);
    v62 = sub_2291EA0(a1, *a2, v6, v31, v32);
    v65 = *(__int64 **)(a1 + 8);
    v33 = sub_DA26C0(v65, (__int64)&v79);
    v34 = sub_DCA690(v65, (__int64)v62, (__int64)v33, 0, 0);
    v35 = sub_DC7ED0(v65, *a2, (__int64)v34, 0, 0);
    *a2 = (__int64)v35;
    *a2 = (__int64)sub_2291F00(a1, (__int64)v35, v6, v36);
    v38 = sub_2291EA0(a1, *a3, v6, (__int64)a2, v37);
    if ( !sub_D968A0((__int64)v38) )
      *a5 = 0;
    goto LABEL_29;
  }
  v26 = sub_228DFC0(a1, 0x20u, v7, v8);
  if ( v26 )
  {
    if ( *(_WORD *)(v7 + 24) || *(_WORD *)(v9 + 24) )
      return 0;
    v61 = v26;
    sub_9865C0((__int64)&v75, *(_QWORD *)(v7 + 32) + 24LL);
    sub_9865C0((__int64)&v77, *(_QWORD *)(v9 + 32) + 24LL);
    sub_C4A3E0((__int64)&v79, (__int64)&v77, (__int64)&v75);
    v63 = sub_2291EA0(a1, *a2, v6, v39, v40);
    v68 = *(__int64 **)(a1 + 8);
    v41 = sub_DA26C0(v68, (__int64)&v79);
    v42 = sub_DCA690(v68, (__int64)v63, (__int64)v41, 0, 0);
    v43 = sub_DC7ED0(v68, *a2, (__int64)v42, 0, 0);
    *a2 = (__int64)v43;
    *a2 = (__int64)sub_2291F00(a1, (__int64)v43, v6, v44);
    v46 = sub_2291FC0(a1, *a3, v6, (__int64)v63, v45);
    *a3 = (__int64)v46;
    v49 = sub_2291EA0(a1, (__int64)v46, v6, v47, v48);
    if ( !sub_D968A0((__int64)v49) )
      *a5 = 0;
LABEL_29:
    sub_969240((__int64 *)&v79);
    sub_969240((__int64 *)&v77);
    sub_969240((__int64 *)&v75);
    return v61;
  }
  v69 = sub_2291EA0(a1, *a2, v6, v27, v28);
  *a2 = (__int64)sub_DCA690(*(__int64 **)(a1 + 8), *a2, v7, 0, 0);
  *a3 = (__int64)sub_DCA690(*(__int64 **)(a1 + 8), *a3, v7, 0, 0);
  v50 = *(__int64 **)(a1 + 8);
  v51 = sub_DCA690(v50, (__int64)v69, v9, 0, 0);
  v52 = sub_DC7ED0(v50, *a2, (__int64)v51, 0, 0);
  *a2 = (__int64)v52;
  *a2 = (__int64)sub_2291F00(a1, (__int64)v52, v6, v53);
  v54 = sub_DCA690(*(__int64 **)(a1 + 8), (__int64)v69, v8, 0, 0);
  v56 = sub_2291FC0(a1, *a3, v6, (__int64)v54, v55);
  *a3 = (__int64)v56;
  v59 = sub_2291EA0(a1, (__int64)v56, v6, v57, v58);
  LOBYTE(v60) = sub_D968A0((__int64)v59);
  v25 = v60;
  if ( !(_BYTE)v60 )
  {
    v25 = 1;
    *a5 = 0;
  }
  return v25;
}
