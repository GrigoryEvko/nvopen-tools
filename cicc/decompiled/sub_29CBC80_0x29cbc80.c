// Function: sub_29CBC80
// Address: 0x29cbc80
//
__int64 __fastcall sub_29CBC80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8,
        char a9)
{
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 *v12; // r14
  unsigned int v13; // r15d
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r14
  __int64 v26; // r14
  _QWORD *v27; // rdi
  __int64 v28; // rbx
  unsigned __int8 v29; // al
  unsigned int v30; // ecx
  __int64 v31; // r15
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // r14
  __int64 *v36; // rbx
  __int64 v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rbx
  __int64 v40; // r14
  const char *v41; // rdx
  __int64 v42; // rax
  const char *v43; // rdi
  __int64 *v44; // rbx
  __int64 v45; // r14
  __int64 v46; // r13
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // r15
  __int64 v50; // [rsp-8h] [rbp-D8h]
  __int64 *v51; // [rsp+18h] [rbp-B8h]
  __int64 v52; // [rsp+20h] [rbp-B0h]
  __int64 v53; // [rsp+30h] [rbp-A0h]
  __int64 v54; // [rsp+30h] [rbp-A0h]
  unsigned int v55; // [rsp+30h] [rbp-A0h]
  _QWORD v56[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v57; // [rsp+60h] [rbp-70h]
  const char *v58; // [rsp+70h] [rbp-60h] BYREF
  __int64 v59; // [rsp+78h] [rbp-58h]
  _QWORD v60[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v61; // [rsp+90h] [rbp-40h]

  v9 = a1;
  v10 = *(_QWORD *)(a1 + 16);
  if ( !v10 )
    goto LABEL_23;
  v11 = sub_B43CC0(a1);
  if ( a9 )
  {
    v36 = *(__int64 **)(a1 + 8);
    v55 = *(_DWORD *)(v11 + 4);
    v61 = 773;
    v58 = sub_BD5D20(a1);
    v60[0] = ".reg2mem";
    v59 = v37;
    v38 = sub_BD2C40(80, 1u);
    v10 = (__int64)v38;
    if ( v38 )
      sub_B4CDD0((__int64)v38, v36, v55, 0, (__int64)&v58, v55, a7, a8);
  }
  else
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *(_DWORD *)(v11 + 4);
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL);
    v58 = sub_BD5D20(a1);
    v61 = 773;
    v59 = v15;
    v60[0] = ".reg2mem";
    v16 = *(_QWORD *)(v14 + 80);
    if ( !v16 )
      BUG();
    v53 = *(_QWORD *)(v16 + 32);
    v17 = sub_BD2C40(80, 1u);
    v10 = (__int64)v17;
    if ( v17 )
    {
      sub_B4CDD0((__int64)v17, v12, v13, 0, (__int64)&v58, v19, v53, 1);
      v18 = v50;
    }
  }
  v20 = 0;
  v54 = 8LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
  {
    do
    {
      v21 = *(_QWORD *)(v9 - 8);
      v22 = *(_QWORD *)(v21 + 4 * v20);
      v23 = *(_QWORD *)(32LL * *(unsigned int *)(v9 + 72) + v21 + v20);
      v24 = *(_QWORD *)(v23 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v24 == v23 + 48 )
      {
        v25 = 0;
      }
      else
      {
        if ( !v24 )
          BUG();
        v25 = v24 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 >= 0xB )
          v25 = 0;
      }
      v26 = v25 + 24;
      v27 = sub_BD2C40(80, unk_3F10A10);
      if ( v27 )
        sub_B4D460((__int64)v27, v22, v10, v26, 0);
      v20 += 8;
    }
    while ( v54 != v20 );
  }
  v28 = v9 + 24;
  while ( 1 )
  {
    v29 = *(_BYTE *)(v28 - 24);
    if ( v29 == 84 )
      goto LABEL_27;
    v30 = v29 - 39;
    if ( v30 > 0x38 || ((1LL << v30) & 0x100060000000001LL) == 0 )
      break;
    if ( v29 == 39 )
      goto LABEL_29;
LABEL_27:
    v28 = *(_QWORD *)(v28 + 8);
    if ( !v28 )
      BUG();
  }
  if ( v29 != 39 )
  {
    v31 = *(_QWORD *)(v9 + 8);
    v58 = sub_BD5D20(v9);
    v60[0] = ".reload";
    v61 = 773;
    v59 = v32;
    v33 = sub_BD2C40(80, 1u);
    v34 = (__int64)v33;
    if ( v33 )
      sub_B4D230((__int64)v33, v31, v10, (__int64)&v58, v28, 0);
    sub_BD84D0(v9, v34);
    goto LABEL_22;
  }
LABEL_29:
  v39 = *(_QWORD *)(v9 + 16);
  v58 = (const char *)v60;
  v59 = 0x400000000LL;
  if ( v39 )
  {
    v40 = *(_QWORD *)(v39 + 24);
    v41 = (const char *)v60;
    v42 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v41[8 * v42] = v40;
      v42 = (unsigned int)(v59 + 1);
      LODWORD(v59) = v59 + 1;
      v39 = *(_QWORD *)(v39 + 8);
      if ( !v39 )
        break;
      v40 = *(_QWORD *)(v39 + 24);
      if ( v42 + 1 > (unsigned __int64)HIDWORD(v59) )
      {
        sub_C8D5F0((__int64)&v58, v60, v42 + 1, 8u, v18, v19);
        v42 = (unsigned int)v59;
      }
      v41 = v58;
    }
    v43 = v58;
    v51 = (__int64 *)&v58[8 * v42];
    if ( v51 != (__int64 *)v58 )
    {
      v52 = v10;
      v44 = (__int64 *)v58;
      do
      {
        v45 = *v44;
        v46 = *(_QWORD *)(v9 + 8);
        v56[0] = sub_BD5D20(v9);
        v56[2] = ".reload";
        v57 = 773;
        v56[1] = v47;
        v48 = sub_BD2C40(80, 1u);
        v49 = (__int64)v48;
        if ( v48 )
          sub_B4D230((__int64)v48, v46, v52, (__int64)v56, v45 + 24, 0);
        ++v44;
        sub_BD2ED0(v45, v9, v49);
      }
      while ( v51 != v44 );
      v10 = v52;
      v43 = v58;
    }
    if ( v43 != (const char *)v60 )
      _libc_free((unsigned __int64)v43);
  }
LABEL_22:
  a1 = v9;
LABEL_23:
  sub_B43D60((_QWORD *)a1);
  return v10;
}
