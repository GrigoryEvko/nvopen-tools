// Function: sub_27E59E0
// Address: 0x27e59e0
//
__int64 __fastcall sub_27E59E0(__int64 *a1, unsigned __int64 a2, __int64 ***a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  _QWORD *v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r13
  unsigned int v13; // r15d
  unsigned __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  char v17; // cl
  __int64 v18; // rdx
  unsigned __int64 v19; // r8
  _QWORD *v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 *v23; // rbx
  unsigned __int64 v24; // rbx
  int v25; // eax
  unsigned __int8 *v26; // rbx
  int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned int v31; // r14d
  int v32; // r15d
  __int64 v33; // rdi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-F8h]
  unsigned __int16 v38; // [rsp+10h] [rbp-F0h]
  char v39; // [rsp+1Ch] [rbp-E4h]
  __int64 v40; // [rsp+20h] [rbp-E0h]
  __int64 *v41; // [rsp+28h] [rbp-D8h]
  __int64 v43; // [rsp+48h] [rbp-B8h]
  unsigned __int8 *v44; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v45; // [rsp+58h] [rbp-A8h]
  __int64 v46; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v47; // [rsp+68h] [rbp-98h]
  __int64 i; // [rsp+70h] [rbp-90h]
  __int64 v49; // [rsp+78h] [rbp-88h]
  __int64 v50; // [rsp+80h] [rbp-80h] BYREF
  _QWORD *v51; // [rsp+88h] [rbp-78h]
  const char *v52; // [rsp+90h] [rbp-70h]
  unsigned int v53; // [rsp+98h] [rbp-68h]
  __int16 v54; // [rsp+A0h] [rbp-60h]
  char v55; // [rsp+C0h] [rbp-40h]

  v4 = (__int64)a1;
  v39 = sub_27DD030((__int64)a1, a2);
  v41 = (__int64 *)sub_27DE210((__int64)a1, v39);
  v7 = sub_27DE090((__int64)a1, v41 != 0);
  v8 = *((unsigned int *)a3 + 2);
  v40 = v7;
  if ( v8 == 1 )
    v43 = (__int64)**a3;
  else
    v43 = sub_27E3B20(a1, a2, *a3, v8, ".thr_comm");
  sub_22C17D0(a1[4], v43, a2, a4);
  v9 = *(_QWORD *)(a2 + 72);
  v50 = (__int64)sub_BD5D20(a2);
  v54 = 773;
  v51 = v10;
  v52 = ".thread";
  v11 = sub_AA48A0(a2);
  v12 = sub_22077B0(0x50u);
  if ( v12 )
    sub_AA4D50(v12, v11, (__int64)&v50, v9, a2);
  sub_AA4AF0(v12, v43);
  if ( v41 )
  {
    v13 = sub_FF0430(v40, v43, a2);
    v50 = sub_FDD860(v41, v43);
    v14 = sub_1098D20((unsigned __int64 *)&v50, v13);
    sub_FE1040(v41, v12, v14);
  }
  v50 = 0;
  v53 = 128;
  v15 = (_QWORD *)sub_C7D670(0x2000, 8);
  v52 = 0;
  v51 = v15;
  v45 = 2;
  v46 = 0;
  v47 = -4096;
  v16 = &v15[8 * (unsigned __int64)v53];
  v44 = (unsigned __int8 *)&unk_49DD7B0;
  for ( i = 0; v16 != v15; v15 += 8 )
  {
    if ( v15 )
    {
      v17 = v45;
      v15[2] = 0;
      v15[3] = -4096;
      *v15 = &unk_49DD7B0;
      v15[1] = v17 & 6;
      v15[4] = i;
    }
  }
  v18 = *(_QWORD *)(a2 + 56);
  v19 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v55 = 0;
  sub_27E2B40((__int64)a1, (__int64)&v50, v18, 1, v19, 0, v12, v43);
  sub_B43C20((__int64)&v44, v12);
  v37 = (__int64)v44;
  v38 = v45;
  v20 = sub_BD2C40(72, 1u);
  if ( v20 )
    sub_B4C8F0((__int64)v20, a4, 1u, v37, v38);
  v21 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v21 )
    goto LABEL_40;
  if ( !v21 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v21 - 24) - 30 > 0xA )
LABEL_40:
    BUG();
  v22 = *(_QWORD *)(v21 + 24);
  v23 = v20 + 6;
  v44 = (unsigned __int8 *)v22;
  if ( !v22 )
  {
    if ( v23 == (unsigned __int64 *)&v44 )
      goto LABEL_20;
    v35 = v20[6];
    if ( !v35 )
      goto LABEL_20;
LABEL_33:
    sub_B91220((__int64)(v20 + 6), v35);
    goto LABEL_34;
  }
  sub_B96E90((__int64)&v44, v22, 1);
  if ( v23 == (unsigned __int64 *)&v44 )
  {
    if ( v44 )
      sub_B91220((__int64)&v44, (__int64)v44);
    goto LABEL_20;
  }
  v35 = v20[6];
  if ( v35 )
    goto LABEL_33;
LABEL_34:
  v36 = v44;
  v20[6] = v44;
  if ( v36 )
    sub_B976B0((__int64)&v44, v36, (__int64)(v20 + 6));
LABEL_20:
  sub_27DBE10(a4, a2, v12, (__int64)&v50);
  v24 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 == v43 + 48 )
  {
    v26 = 0;
  }
  else
  {
    if ( !v24 )
      BUG();
    v25 = *(unsigned __int8 *)(v24 - 24);
    v26 = (unsigned __int8 *)(v24 - 24);
    if ( (unsigned int)(v25 - 30) >= 0xB )
      v26 = 0;
  }
  v27 = sub_B46E30((__int64)v26);
  if ( v27 )
  {
    v31 = 0;
    v32 = v27;
    do
    {
      while ( a2 != sub_B46EC0((__int64)v26, v31) )
      {
        if ( v32 == ++v31 )
          goto LABEL_29;
      }
      sub_AA5980(a2, v43, 1u);
      sub_B46F90(v26, v31++, v12);
    }
    while ( v32 != v31 );
LABEL_29:
    v4 = (__int64)a1;
  }
  v33 = *(_QWORD *)(v4 + 48);
  v44 = (unsigned __int8 *)v12;
  v46 = v43;
  v45 = a4 & 0xFFFFFFFFFFFFFFFBLL;
  i = v43;
  v47 = v12 & 0xFFFFFFFFFFFFFFFBLL;
  v49 = a2 | 4;
  sub_FFDB80(v33, (unsigned __int64 *)&v44, 3, v28, v29, v30);
  sub_27E1050(v4, a2, v12, (__int64)&v50);
  sub_F61E50(v12, *(__int64 **)(v4 + 16));
  sub_27DE490(v4, v43, a2, v12, a4, v41, v40, v39);
  return sub_27DCE00((__int64)&v50);
}
