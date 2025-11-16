// Function: sub_2700B00
// Address: 0x2700b00
//
void __fastcall sub_2700B00(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        _DWORD *a5,
        __int64 a6,
        _BYTE *a7,
        unsigned __int64 a8)
{
  __int64 v12; // r13
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  bool v15; // zf
  __int64 *v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r12
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // r10
  __int64 v30; // rbx
  _QWORD *v31; // rax
  unsigned __int16 v32; // r9
  __int64 v33; // r13
  unsigned __int16 v34; // bx
  __int64 v35; // r12
  _QWORD *v36; // rdi
  __int64 v37; // rdi
  _QWORD **v38; // r13
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-118h]
  __int64 v41; // [rsp+18h] [rbp-108h]
  __int64 v42; // [rsp+18h] [rbp-108h]
  _QWORD **v43; // [rsp+20h] [rbp-100h]
  unsigned __int16 v44; // [rsp+20h] [rbp-100h]
  __int64 v45; // [rsp+20h] [rbp-100h]
  __int64 v46; // [rsp+20h] [rbp-100h]
  unsigned int v47; // [rsp+28h] [rbp-F8h]
  char v48; // [rsp+28h] [rbp-F8h]
  __int64 *v49; // [rsp+28h] [rbp-F8h]
  unsigned int v50; // [rsp+28h] [rbp-F8h]
  __int64 v53; // [rsp+40h] [rbp-E0h]
  __int64 v54; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int16 v55; // [rsp+58h] [rbp-C8h]
  __int64 *v56; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+68h] [rbp-B8h]
  __int64 v58; // [rsp+70h] [rbp-B0h] BYREF
  const char *v59; // [rsp+80h] [rbp-A0h] BYREF
  unsigned __int16 v60; // [rsp+88h] [rbp-98h]
  __int16 v61; // [rsp+A0h] [rbp-80h]
  _QWORD *v62; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD v63[12]; // [rsp+C0h] [rbp-60h] BYREF

  v12 = *a1;
  v13 = *(_BYTE **)(*a1 + 232);
  v14 = (__int64)&v13[*(_QWORD *)(*a1 + 240)];
  v62 = v63;
  sub_26F69E0((__int64 *)&v62, v13, v14);
  v15 = *(_DWORD *)(v12 + 264) == 39;
  v63[2] = *(_QWORD *)(v12 + 264);
  v63[3] = *(_QWORD *)(v12 + 272);
  v63[4] = *(_QWORD *)(v12 + 280);
  if ( !v15 || (unsigned int)qword_4FF9568 < a3 )
    goto LABEL_27;
  if ( *(_BYTE *)(a4 + 24) )
  {
    v37 = *(_QWORD *)(a4 + 104);
    if ( v37 == a4 + 88 )
      goto LABEL_27;
    while ( *(_BYTE *)(v37 + 80) )
    {
      v37 = sub_220EEE0(v37);
      if ( v37 == a4 + 88 )
        goto LABEL_27;
    }
  }
  v59 = (const char *)a1[8];
  v16 = (__int64 *)sub_BCB120(*(_QWORD **)*a1);
  v17 = sub_BCF480(v16, &v59, 1, 1u);
  if ( *a7 )
  {
    v38 = (_QWORD **)*a1;
    v46 = v17;
    v59 = "branch_funnel";
    v61 = 259;
    v50 = *((_DWORD *)v38 + 80);
    v39 = sub_BD2DA0(136);
    v53 = v39;
    if ( v39 )
      sub_B2C3B0(v39, v46, 7, v50, (__int64)&v59, (__int64)v38);
  }
  else
  {
    v41 = v17;
    v43 = (_QWORD **)*a1;
    sub_26F78E0((__int64 *)&v56, (__int64)a7, a8, 0, 0, *a1, "branch_funnel", 0xDu);
    v18 = *a1;
    v59 = (const char *)&v56;
    v61 = 260;
    v47 = *(_DWORD *)(v18 + 320);
    v19 = sub_BD2DA0(136);
    v53 = v19;
    if ( v19 )
      sub_B2C3B0(v19, v41, 0, v47, (__int64)&v59, (__int64)v43);
    if ( v56 != &v58 )
      j_j___libc_free_0((unsigned __int64)v56);
    v48 = *(_BYTE *)(v53 + 32);
    *(_BYTE *)(v53 + 32) = v48 & 0xCF | 0x10;
    if ( (v48 & 0xF) != 9 )
      *(_BYTE *)(v53 + 33) |= 0x40u;
  }
  sub_B2D3C0(v53, 0, 21);
  v56 = 0;
  v57 = 0;
  v58 = 0;
  if ( (*(_BYTE *)(v53 + 2) & 1) != 0 )
    sub_B2C6D0(v53, 0, v20, v21);
  v22 = a2 + 32 * a3;
  v59 = *(const char **)(v53 + 96);
  sub_26F6E90((__int64)&v56, &v59);
  while ( v22 != a2 )
  {
    v23 = *(_QWORD *)(a2 + 8);
    a2 += 32;
    v59 = (const char *)sub_26F6BC0((__int64)a1, v23);
    sub_26F6E90((__int64)&v56, &v59);
    v59 = *(const char **)(a2 - 32);
    sub_26F6E90((__int64)&v56, &v59);
  }
  v24 = (__int64 *)*a1;
  v61 = 257;
  v25 = *v24;
  v26 = sub_22077B0(0x50u);
  v27 = v26;
  if ( v26 )
    sub_AA4D50(v26, v25, (__int64)&v59, v53, 0);
  v28 = sub_B6E160((__int64 *)*a1, 0xC2u, 0, 0);
  sub_B43C20((__int64)&v54, v27);
  v29 = 0;
  v61 = 257;
  v49 = v56;
  v30 = (v57 - (__int64)v56) >> 3;
  if ( v28 )
    v29 = *(_QWORD *)(v28 + 24);
  v42 = v29;
  v40 = v54;
  v44 = v55;
  v31 = sub_BD2C40(88, (int)v30 + 1);
  if ( v31 )
  {
    v32 = v44;
    v45 = (__int64)v31;
    sub_B44260((__int64)v31, **(_QWORD **)(v42 + 16), 56, (v30 + 1) & 0x7FFFFFF, v40, v32);
    *(_QWORD *)(v45 + 72) = 0;
    sub_B4A290(v45, v42, v28, v49, v30, (__int64)&v59, 0, 0);
    v31 = (_QWORD *)v45;
  }
  *((_WORD *)v31 + 1) = *((_WORD *)v31 + 1) & 0xFFFC | 2;
  sub_B43C20((__int64)&v59, v27);
  v33 = (__int64)v59;
  v34 = v60;
  v35 = *(_QWORD *)*a1;
  v36 = sub_BD2C40(72, 0);
  if ( v36 )
    sub_B4BB80((__int64)v36, v35, 0, 0, v33, v34);
  LOBYTE(v59) = 0;
  sub_2700A50((__int64)a1, a4, v53, &v59);
  if ( (_BYTE)v59 )
    *a5 = 2;
  if ( v56 )
    j_j___libc_free_0((unsigned __int64)v56);
LABEL_27:
  if ( v62 != v63 )
    j_j___libc_free_0((unsigned __int64)v62);
}
