// Function: ctor_078_0
// Address: 0x49c8e0
//
int ctor_078_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int *v12; // rbx
  int *v13; // rdi
  int *v14; // rbx
  int *v15; // rdi
  char v17; // [rsp+6Fh] [rbp-271h] BYREF
  _DWORD v18[2]; // [rsp+70h] [rbp-270h] BYREF
  __int64 v19; // [rsp+78h] [rbp-268h] BYREF
  int v20; // [rsp+80h] [rbp-260h] BYREF
  int v21; // [rsp+84h] [rbp-25Ch]
  unsigned __int64 v22; // [rsp+88h] [rbp-258h] BYREF
  _QWORD v23[2]; // [rsp+90h] [rbp-250h] BYREF
  char v24; // [rsp+A0h] [rbp-240h] BYREF
  _QWORD v25[2]; // [rsp+D0h] [rbp-210h] BYREF
  _BYTE v26[48]; // [rsp+E0h] [rbp-200h] BYREF
  _QWORD v27[2]; // [rsp+110h] [rbp-1D0h] BYREF
  _BYTE v28[48]; // [rsp+120h] [rbp-1C0h] BYREF
  _QWORD v29[2]; // [rsp+150h] [rbp-190h] BYREF
  _BYTE v30[48]; // [rsp+160h] [rbp-180h] BYREF
  int v31; // [rsp+190h] [rbp-150h] BYREF
  _QWORD v32[2]; // [rsp+198h] [rbp-148h] BYREF
  _BYTE v33[48]; // [rsp+1A8h] [rbp-138h] BYREF
  int v34; // [rsp+1D8h] [rbp-108h]
  _QWORD v35[2]; // [rsp+1E0h] [rbp-100h] BYREF
  _BYTE v36[48]; // [rsp+1F0h] [rbp-F0h] BYREF
  int v37; // [rsp+220h] [rbp-C0h]
  _BYTE v38[64]; // [rsp+228h] [rbp-B8h] BYREF
  int v39; // [rsp+268h] [rbp-78h] BYREF
  char v40[64]; // [rsp+270h] [rbp-70h] BYREF
  char v41; // [rsp+2B0h] [rbp-30h] BYREF

  qword_4F8E920 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8E970 = 0x100000000LL;
  dword_4F8E92C &= 0x8000u;
  word_4F8E930 = 0;
  qword_4F8E938 = 0;
  qword_4F8E940 = 0;
  dword_4F8E928 = v0;
  qword_4F8E948 = 0;
  qword_4F8E950 = 0;
  qword_4F8E958 = 0;
  qword_4F8E960 = 0;
  qword_4F8E968 = (__int64)&unk_4F8E978;
  qword_4F8E980 = 0;
  qword_4F8E988 = (__int64)&unk_4F8E9A0;
  qword_4F8E990 = 1;
  dword_4F8E998 = 0;
  byte_4F8E99C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8E970;
  v3 = (unsigned int)qword_4F8E970 + 1LL;
  if ( v3 > HIDWORD(qword_4F8E970) )
  {
    sub_C8D5F0((char *)&unk_4F8E978 - 16, &unk_4F8E978, v3, 8);
    v2 = (unsigned int)qword_4F8E970;
  }
  *(_QWORD *)(qword_4F8E968 + 8 * v2) = v1;
  LODWORD(qword_4F8E970) = qword_4F8E970 + 1;
  qword_4F8E9A8 = 0;
  qword_4F8E9B0 = (__int64)&unk_49D9748;
  qword_4F8E9B8 = 0;
  qword_4F8E920 = (__int64)&unk_49DC090;
  qword_4F8E9C0 = (__int64)&unk_49DC1D0;
  qword_4F8E9E0 = (__int64)nullsub_23;
  qword_4F8E9D8 = (__int64)sub_984030;
  sub_C53080(&qword_4F8E920, "print-bpi", 9);
  LOWORD(qword_4F8E9B8) = 256;
  LOBYTE(qword_4F8E9A8) = 0;
  qword_4F8E950 = 34;
  LOBYTE(dword_4F8E92C) = dword_4F8E92C & 0x9F | 0x20;
  qword_4F8E948 = (__int64)"Print the branch probability info.";
  sub_C53130(&qword_4F8E920);
  __cxa_atexit(sub_984900, &qword_4F8E920, &qword_4A427C0);
  qword_4F8E820 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8E870 = 0x100000000LL;
  word_4F8E830 = 0;
  dword_4F8E82C &= 0x8000u;
  qword_4F8E838 = 0;
  qword_4F8E840 = 0;
  dword_4F8E828 = v4;
  qword_4F8E848 = 0;
  qword_4F8E850 = 0;
  qword_4F8E858 = 0;
  qword_4F8E860 = 0;
  qword_4F8E868 = (__int64)&unk_4F8E878;
  qword_4F8E880 = 0;
  qword_4F8E888 = (__int64)&unk_4F8E8A0;
  qword_4F8E890 = 1;
  dword_4F8E898 = 0;
  byte_4F8E89C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8E870;
  v7 = (unsigned int)qword_4F8E870 + 1LL;
  if ( v7 > HIDWORD(qword_4F8E870) )
  {
    sub_C8D5F0((char *)&unk_4F8E878 - 16, &unk_4F8E878, v7, 8);
    v6 = (unsigned int)qword_4F8E870;
  }
  *(_QWORD *)(qword_4F8E868 + 8 * v6) = v5;
  qword_4F8E8A8 = &byte_4F8E8B8;
  qword_4F8E8D0 = (__int64)&byte_4F8E8E0;
  LODWORD(qword_4F8E870) = qword_4F8E870 + 1;
  qword_4F8E8B0 = 0;
  qword_4F8E8C8 = (__int64)&unk_49DC130;
  byte_4F8E8B8 = 0;
  byte_4F8E8E0 = 0;
  qword_4F8E820 = (__int64)&unk_49DC010;
  qword_4F8E8D8 = 0;
  byte_4F8E8F0 = 0;
  qword_4F8E8F8 = (__int64)&unk_49DC350;
  qword_4F8E918 = (__int64)nullsub_92;
  qword_4F8E910 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4F8E820, "print-bpi-func-name", 19);
  qword_4F8E850 = 88;
  LOBYTE(dword_4F8E82C) = dword_4F8E82C & 0x9F | 0x20;
  qword_4F8E848 = (__int64)"The option to specify the name of the function whose branch probability info is printed.";
  sub_C53130(&qword_4F8E820);
  __cxa_atexit(sub_BC5A40, &qword_4F8E820, &qword_4A427C0);
  qword_4F8E740 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8E74C &= 0x8000u;
  word_4F8E750 = 0;
  qword_4F8E790 = 0x100000000LL;
  qword_4F8E758 = 0;
  qword_4F8E760 = 0;
  qword_4F8E768 = 0;
  dword_4F8E748 = v8;
  qword_4F8E770 = 0;
  qword_4F8E778 = 0;
  qword_4F8E780 = 0;
  qword_4F8E788 = (__int64)&unk_4F8E798;
  qword_4F8E7A0 = 0;
  qword_4F8E7A8 = (__int64)&unk_4F8E7C0;
  qword_4F8E7B0 = 1;
  dword_4F8E7B8 = 0;
  byte_4F8E7BC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F8E790;
  v11 = (unsigned int)qword_4F8E790 + 1LL;
  if ( v11 > HIDWORD(qword_4F8E790) )
  {
    sub_C8D5F0((char *)&unk_4F8E798 - 16, &unk_4F8E798, v11, 8);
    v10 = (unsigned int)qword_4F8E790;
  }
  *(_QWORD *)(qword_4F8E788 + 8 * v10) = v9;
  LODWORD(qword_4F8E790) = qword_4F8E790 + 1;
  qword_4F8E7C8 = 0;
  qword_4F8E7D0 = (__int64)&unk_49D9728;
  qword_4F8E7D8 = 0;
  qword_4F8E740 = (__int64)&unk_49DBF10;
  qword_4F8E7E0 = (__int64)&unk_49DC290;
  qword_4F8E800 = (__int64)nullsub_24;
  qword_4F8E7F8 = (__int64)sub_984050;
  sub_C53080(&qword_4F8E740, "branch-prob-lbh-trip-count", 26);
  LODWORD(qword_4F8E7C8) = 31;
  BYTE4(qword_4F8E7D8) = 1;
  LODWORD(qword_4F8E7D8) = 31;
  qword_4F8E770 = 71;
  LOBYTE(dword_4F8E74C) = dword_4F8E74C & 0x9F | 0x20;
  qword_4F8E768 = (__int64)"Assumed number of iterations a loop will take for branch-prob analysis.";
  sub_C53130(&qword_4F8E740);
  __cxa_atexit(sub_984970, &qword_4F8E740, &qword_4A427C0);
  dword_4F8E738 = 1;
  sub_F02DB0((char *)&qword_4F8E730 + 4, 20, 32);
  sub_F02DB0(&qword_4F8E730, 12, 32);
  v25[0] = __PAIR64__(qword_4F8E730, HIDWORD(qword_4F8E730));
  sub_FF1140(v29, v25, 2);
  v31 = 33;
  sub_FF11D0(v32, v29);
  v23[0] = qword_4F8E730;
  sub_FF1140(v27, v23, 2);
  v34 = 32;
  sub_FF11D0(v35, v27);
  sub_FF1A00(&unk_4F8E700, &v31, 2, &v22);
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0], &v31);
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0], &v31);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  __cxa_atexit(sub_FEE6C0, &unk_4F8E700, &qword_4A427C0);
  sub_F02DB0((char *)&qword_4F8E6F0 + 4, 20, 32);
  sub_F02DB0(&qword_4F8E6F0, 12, 32);
  v22 = qword_4F8E6F0;
  sub_FF1140(v29, &v22, 2);
  v31 = 32;
  sub_FF11D0(v32, v29);
  v20 = HIDWORD(qword_4F8E6F0);
  v21 = qword_4F8E6F0;
  sub_FF1140(v27, &v20, 2);
  v34 = 33;
  sub_FF11D0(v35, v27);
  v19 = qword_4F8E6F0;
  sub_FF1140(v25, &v19, 2);
  v37 = 40;
  sub_FF11D0(v38, v25);
  v18[0] = HIDWORD(qword_4F8E6F0);
  v18[1] = qword_4F8E6F0;
  sub_FF1140(v23, v18, 2);
  v39 = 38;
  sub_FF11D0(v40, v23);
  sub_FF1A00(&unk_4F8E6C0, &v31, 4, &v17);
  v12 = (int *)&v41;
  do
  {
    v12 -= 18;
    v13 = (int *)*((_QWORD *)v12 + 1);
    if ( v13 != v12 + 6 )
      _libc_free(v13, &v31);
  }
  while ( v12 != &v31 );
  if ( (char *)v23[0] != &v24 )
    _libc_free(v23[0], &v31);
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0], &v31);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  __cxa_atexit(sub_FEE6C0, &unk_4F8E6C0, &qword_4A427C0);
  v23[0] = qword_4F8E6F0;
  sub_FF1140(v29, v23, 2);
  v31 = 32;
  sub_FF11D0(v32, v29);
  v22 = __PAIR64__(qword_4F8E6F0, HIDWORD(qword_4F8E6F0));
  sub_FF1140(v27, &v22, 2);
  v34 = 33;
  sub_FF11D0(v35, v27);
  v20 = HIDWORD(qword_4F8E6F0);
  v21 = qword_4F8E6F0;
  sub_FF1140(v25, &v20, 2);
  v37 = 38;
  sub_FF11D0(v38, v25);
  v14 = &v39;
  sub_FF1A00(&unk_4F8E680, &v31, 3, &v19);
  do
  {
    v14 -= 18;
    v15 = (int *)*((_QWORD *)v14 + 1);
    if ( v15 != v14 + 6 )
      _libc_free(v15, &v31);
  }
  while ( v14 != &v31 );
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0], &v31);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  __cxa_atexit(sub_FEE6C0, &unk_4F8E680, &qword_4A427C0);
  v27[0] = qword_4F8E6F0;
  sub_FF1140(v29, v27, 2);
  v31 = 40;
  sub_FF11D0(v32, v29);
  sub_FF1A00(&unk_4F8E640, &v31, 1, v25);
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  __cxa_atexit(sub_FEE6C0, &unk_4F8E640, &qword_4A427C0);
  v25[0] = qword_4F8E6F0;
  sub_FF1140(v29, v25, 2);
  v31 = 32;
  sub_FF11D0(v32, v29);
  v23[0] = __PAIR64__(qword_4F8E6F0, HIDWORD(qword_4F8E6F0));
  sub_FF1140(v27, v23, 2);
  v34 = 33;
  sub_FF11D0(v35, v27);
  sub_FF1A00(&unk_4F8E600, &v31, 2, &v22);
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0], &v31);
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0], &v31);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  __cxa_atexit(sub_FEE6C0, &unk_4F8E600, &qword_4A427C0);
  sub_F02DB0((char *)&qword_4F8E5F8 + 4, 0xFFFFF, 0x100000);
  sub_F02DB0(&qword_4F8E5F8, 1, 0x100000);
  sub_F02DB0(&dword_4F8E5F4, 20, 32);
  sub_F02DB0(&dword_4F8E5F0, 12, 32);
  v25[0] = __PAIR64__(qword_4F8E5F8, HIDWORD(qword_4F8E5F8));
  sub_FF1140(v29, v25, 2);
  v31 = 7;
  sub_FF11D0(v32, v29);
  v23[0] = qword_4F8E5F8;
  sub_FF1140(v27, v23, 2);
  v34 = 8;
  sub_FF11D0(v35, v27);
  sub_FF1A00(&unk_4F8E5C0, &v31, 2, &v22);
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0], &v31);
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0], &v31);
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0], &v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], &v31);
  return __cxa_atexit(sub_FEE6C0, &unk_4F8E5C0, &qword_4A427C0);
}
