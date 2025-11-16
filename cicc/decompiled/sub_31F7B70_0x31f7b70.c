// Function: sub_31F7B70
// Address: 0x31f7b70
//
void __fastcall sub_31F7B70(__int64 a1)
{
  __int64 v1; // r13
  void (__fastcall *v2)(__int64, _QWORD, _QWORD); // rbx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rax
  unsigned __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  char v17; // edx^3
  unsigned __int64 v18; // rax
  __int64 *v19; // r13
  __int64 *v20; // rbx
  __int64 *i; // rax
  __int64 v22; // rdi
  unsigned int v23; // ecx
  __int64 v24; // rsi
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rsi
  __int64 v28; // rdi
  unsigned int v29; // [rsp+Bh] [rbp-1C5h]
  unsigned __int64 v30; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v31; // [rsp+18h] [rbp-1B8h] BYREF
  _QWORD v32[2]; // [rsp+20h] [rbp-1B0h] BYREF
  _QWORD v33[4]; // [rsp+30h] [rbp-1A0h] BYREF
  void *v34; // [rsp+50h] [rbp-180h] BYREF
  _QWORD *v35; // [rsp+58h] [rbp-178h]
  _QWORD *v36; // [rsp+60h] [rbp-170h]
  _QWORD *v37; // [rsp+68h] [rbp-168h]
  char *v38; // [rsp+70h] [rbp-160h]
  __int64 v39; // [rsp+90h] [rbp-140h]
  void *v40; // [rsp+A0h] [rbp-130h] BYREF
  char v41; // [rsp+AAh] [rbp-126h]
  char v42; // [rsp+AEh] [rbp-122h]
  _BYTE *v43; // [rsp+B0h] [rbp-120h]
  __int64 v44; // [rsp+B8h] [rbp-118h]
  _BYTE v45[24]; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v46; // [rsp+D8h] [rbp-F8h]
  __int64 v47; // [rsp+E0h] [rbp-F0h]
  _QWORD *v48; // [rsp+E8h] [rbp-E8h]
  __int64 v49; // [rsp+F0h] [rbp-E0h]
  _QWORD v50[3]; // [rsp+100h] [rbp-D0h] BYREF
  __int64 *v51; // [rsp+118h] [rbp-B8h]
  int v52; // [rsp+120h] [rbp-B0h]
  char v53; // [rsp+128h] [rbp-A8h] BYREF
  __int64 *v54; // [rsp+148h] [rbp-88h]
  unsigned int v55; // [rsp+150h] [rbp-80h]
  char v56; // [rsp+158h] [rbp-78h] BYREF
  unsigned __int64 v57; // [rsp+170h] [rbp-60h]

  v1 = *(_QWORD *)(a1 + 528);
  v2 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v1 + 176LL);
  v3 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v2(v1, *(_QWORD *)(v3 + 400), 0);
  sub_31F7AE0(a1);
  v4 = sub_3707D00(a1 + 632);
  sub_37175A0(v50, v4);
  v35 = 0;
  v33[0] = &off_49D8C58;
  v34 = &unk_4A35300;
  v5 = *(_QWORD *)(a1 + 528);
  v36 = 0;
  v33[1] = v5;
  v37 = 0;
  v33[2] = v50;
  v40 = &unk_4A3C998;
  v44 = 0x200000000LL;
  v41 = 0;
  v42 = 0;
  v43 = v45;
  v46 = 0;
  v47 = 0;
  v48 = v33;
  v49 = 0;
  v6 = (_QWORD *)sub_22077B0(8u);
  if ( v6 )
    *v6 = &v40;
  v35 = v6;
  v7 = &v30;
  v36 = v6 + 1;
  v37 = v6 + 1;
  v8 = sub_37173B0(v50);
  v29 = v8;
  if ( BYTE4(v8) )
  {
    while ( 1 )
    {
      v9 = sub_3717160(v50, v29);
      v32[1] = v10;
      v32[0] = v9;
      sub_3707360(&v30, v32, v29, &v34, 0);
      if ( (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      v11 = v29;
      v30 = 0;
      v16 = sub_3717410(v50, v29);
      LOWORD(v29) = v16;
      BYTE2(v29) = BYTE2(v16);
      v17 = BYTE3(v16);
      v18 = HIDWORD(v16);
      HIBYTE(v29) = v17;
      if ( (v30 & 1) != 0 || (v30 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_7;
      if ( !(_BYTE)v18 )
        goto LABEL_11;
    }
    v7 = (unsigned __int64 *)&v31;
    v30 = v30 & 0xFFFFFFFFFFFFFFFELL | 1;
    v38 = "error: ";
    LOWORD(v39) = 259;
    v11 = (__int64)sub_CB72A0();
    v31 = v30 | 1;
    v30 = 0;
    sub_C63F70((unsigned __int64 *)&v31, (__int64 *)v11, v12, v13, v14, v15, (char)v38);
    if ( (v31 & 1) == 0 && (v31 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      BUG();
LABEL_7:
    sub_C63C30(v7, v11);
  }
LABEL_11:
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  v34 = &unk_4A35300;
  if ( v35 )
    j_j___libc_free_0((unsigned __int64)v35);
  v50[0] = &unk_4A3CB08;
  if ( v57 )
    j_j___libc_free_0(v57);
  v19 = v51;
  v20 = &v51[v52];
  if ( v51 != v20 )
  {
    for ( i = v51; ; i = v51 )
    {
      v22 = *v19;
      v23 = (unsigned int)(v19 - i) >> 7;
      v24 = 4096LL << v23;
      if ( v23 >= 0x1E )
        v24 = 0x40000000000LL;
      ++v19;
      sub_C7D6A0(v22, v24, 16);
      if ( v20 == v19 )
        break;
    }
  }
  v25 = v54;
  v26 = &v54[2 * v55];
  if ( v54 != v26 )
  {
    do
    {
      v27 = v25[1];
      v28 = *v25;
      v25 += 2;
      sub_C7D6A0(v28, v27, 16);
    }
    while ( v26 != v25 );
    v26 = v54;
  }
  if ( v26 != (__int64 *)&v56 )
    _libc_free((unsigned __int64)v26);
  if ( v51 != (__int64 *)&v53 )
    _libc_free((unsigned __int64)v51);
}
