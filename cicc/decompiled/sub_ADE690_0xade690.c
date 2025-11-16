// Function: sub_ADE690
// Address: 0xade690
//
__int64 __fastcall sub_ADE690(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r11
  __int64 v9; // r10
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned __int64 v15; // r12
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // r11
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  unsigned int *v29; // rax
  int v30; // ecx
  unsigned int *v31; // rdx
  __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-138h]
  __int64 v39; // [rsp+10h] [rbp-130h]
  __int64 v40; // [rsp+10h] [rbp-130h]
  __int64 v41; // [rsp+10h] [rbp-130h]
  _QWORD v43[4]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v44; // [rsp+40h] [rbp-100h]
  _QWORD v45[6]; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int *v46; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+88h] [rbp-B8h]
  _BYTE v48[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-90h]
  __int64 v50; // [rsp+B8h] [rbp-88h]
  __int16 v51; // [rsp+C0h] [rbp-80h]
  __int64 v52; // [rsp+C8h] [rbp-78h]
  void **v53; // [rsp+D0h] [rbp-70h]
  _QWORD *v54; // [rsp+D8h] [rbp-68h]
  __int64 v55; // [rsp+E0h] [rbp-60h]
  int v56; // [rsp+E8h] [rbp-58h]
  __int16 v57; // [rsp+ECh] [rbp-54h]
  char v58; // [rsp+EEh] [rbp-52h]
  __int64 v59; // [rsp+F0h] [rbp-50h]
  __int64 v60; // [rsp+F8h] [rbp-48h]
  void *v61; // [rsp+100h] [rbp-40h] BYREF
  _QWORD v62[7]; // [rsp+108h] [rbp-38h] BYREF

  v8 = a3;
  v9 = 0;
  v12 = a2;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    a2 = 38;
    v13 = sub_B91C10(v12, 38);
    v8 = a3;
    v9 = v13;
  }
  if ( *(_BYTE *)(*a1 + 872LL) )
  {
    v14 = sub_B128C0(v8, a4, a5, v9, a6, a7, a8);
    sub_ADE610((__int64)a1, v14, *(_QWORD *)(v12 + 32), 1u);
    return v14 | 4;
  }
  v37 = v8;
  v39 = v9;
  v17 = sub_BD5C60(v12, a2, a3);
  v18 = sub_B43CA0(v12);
  v21 = v39;
  v22 = v37;
  if ( !a1[6] )
  {
    a2 = 68;
    v35 = sub_B6E160(v18, 68, 0, 0);
    v22 = v37;
    v21 = v39;
    a1[6] = v35;
  }
  v40 = v21;
  v23 = sub_B98A20(v22, a2, v19, v20);
  v45[0] = sub_B9F6F0(v17, v23);
  v45[1] = sub_B9F6F0(v17, a4);
  v45[2] = sub_B9F6F0(v17, a5);
  v45[3] = sub_B9F6F0(v17, v40);
  v26 = sub_B98A20(a6, v40, v24, v25);
  v45[4] = sub_B9F6F0(v17, v26);
  v27 = sub_B9F6F0(v17, a7);
  v52 = v17;
  v51 = 0;
  v45[5] = v27;
  v47 = 0x200000000LL;
  v57 = 512;
  v61 = &unk_49DA100;
  v46 = (unsigned int *)v48;
  v53 = &v61;
  v54 = v62;
  v55 = 0;
  v56 = 0;
  v58 = 7;
  v59 = 0;
  v60 = 0;
  v49 = 0;
  v50 = 0;
  v62[0] = &unk_49DA0B0;
  sub_B10CB0(v43, a8);
  v28 = v43[0];
  if ( !v43[0] )
  {
    sub_93FB40((__int64)&v46, 0);
    v28 = v43[0];
    goto LABEL_23;
  }
  v29 = v46;
  v30 = v47;
  v31 = &v46[4 * (unsigned int)v47];
  if ( v46 == v31 )
  {
LABEL_19:
    if ( (unsigned int)v47 >= (unsigned __int64)HIDWORD(v47) )
    {
      v36 = (unsigned int)v47 + 1LL;
      if ( HIDWORD(v47) < v36 )
      {
        v41 = v43[0];
        sub_C8D5F0(&v46, v48, v36, 16);
        v28 = v41;
        v31 = &v46[4 * (unsigned int)v47];
      }
      *(_QWORD *)v31 = 0;
      *((_QWORD *)v31 + 1) = v28;
      v28 = v43[0];
      LODWORD(v47) = v47 + 1;
    }
    else
    {
      if ( v31 )
      {
        *v31 = 0;
        *((_QWORD *)v31 + 1) = v28;
        v30 = v47;
        v28 = v43[0];
      }
      LODWORD(v47) = v30 + 1;
    }
LABEL_23:
    if ( !v28 )
      goto LABEL_15;
    goto LABEL_14;
  }
  while ( *v29 )
  {
    v29 += 4;
    if ( v31 == v29 )
      goto LABEL_19;
  }
  *((_QWORD *)v29 + 1) = v43[0];
LABEL_14:
  sub_B91220(v43);
LABEL_15:
  v32 = a1[6];
  v33 = 0;
  v44 = 257;
  if ( v32 )
    v33 = *(_QWORD *)(v32 + 24);
  v34 = sub_921880(&v46, v33, v32, (int)v45, 6, (__int64)v43, 0);
  sub_B43E90(v34, v12 + 24, 0);
  v15 = v34 & 0xFFFFFFFFFFFFFFFBLL;
  nullsub_61(v62);
  v61 = &unk_49DA100;
  nullsub_63(&v61);
  if ( v46 != (unsigned int *)v48 )
    _libc_free(v46, v12 + 24);
  return v15;
}
