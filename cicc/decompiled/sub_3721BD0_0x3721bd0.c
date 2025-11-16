// Function: sub_3721BD0
// Address: 0x3721bd0
//
__int64 __fastcall sub_3721BD0(unsigned __int16 *a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdi
  void (*v5)(); // rax
  __int64 v6; // rdi
  void (*v7)(); // rax
  __int64 v8; // rdi
  void (*v9)(); // rax
  __int64 v10; // rdi
  void (*v11)(); // rax
  __int64 v12; // rdi
  void (*v13)(); // rax
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // rdi
  void (*v17)(); // rax
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // rdi
  void (*v21)(); // rax
  __int64 v22; // rdi
  void (*v23)(); // rax
  _QWORD v25[4]; // [rsp+30h] [rbp-60h] BYREF
  char v26; // [rsp+50h] [rbp-40h]
  char v27; // [rsp+51h] [rbp-3Fh]

  v3 = *a2;
  v25[0] = "Header: unit length";
  v27 = 1;
  v26 = 3;
  a2[35] = sub_31F0F50(v3);
  v4 = *(_QWORD *)(v3 + 224);
  v5 = *(void (**)())(*(_QWORD *)v4 + 120LL);
  v27 = 1;
  v25[0] = "Header: version";
  v26 = 3;
  if ( v5 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v5)(v4, v25, 1);
  sub_31DC9F0(v3, *a1);
  v6 = *(_QWORD *)(v3 + 224);
  v7 = *(void (**)())(*(_QWORD *)v6 + 120LL);
  v27 = 1;
  v25[0] = "Header: padding";
  v26 = 3;
  if ( v7 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v7)(v6, v25, 1);
  sub_31DC9F0(v3, a1[1]);
  v8 = *(_QWORD *)(v3 + 224);
  v9 = *(void (**)())(*(_QWORD *)v8 + 120LL);
  v27 = 1;
  v25[0] = "Header: compilation unit count";
  v26 = 3;
  if ( v9 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v9)(v8, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 1));
  v10 = *(_QWORD *)(v3 + 224);
  v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
  v27 = 1;
  v25[0] = "Header: local type unit count";
  v26 = 3;
  if ( v11 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v11)(v10, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 2));
  v12 = *(_QWORD *)(v3 + 224);
  v13 = *(void (**)())(*(_QWORD *)v12 + 120LL);
  v27 = 1;
  v25[0] = "Header: foreign type unit count";
  v26 = 3;
  if ( v13 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v13)(v12, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 3));
  v14 = *(_QWORD *)(v3 + 224);
  v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
  v27 = 1;
  v25[0] = "Header: bucket count";
  v26 = 3;
  if ( v15 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v14, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 4));
  v16 = *(_QWORD *)(v3 + 224);
  v17 = *(void (**)())(*(_QWORD *)v16 + 120LL);
  v27 = 1;
  v25[0] = "Header: name count";
  v26 = 3;
  if ( v17 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v16, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 5));
  v18 = *(_QWORD *)(v3 + 224);
  v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
  v27 = 1;
  v25[0] = "Header: abbreviation table size";
  v26 = 3;
  if ( v19 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v19)(v18, v25, 1);
  sub_31DCA50(v3);
  v20 = *(_QWORD *)(v3 + 224);
  v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
  v27 = 1;
  v25[0] = "Header: augmentation string size";
  v26 = 3;
  if ( v21 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v25, 1);
  sub_31DCA10(v3, *((_DWORD *)a1 + 7));
  v22 = *(_QWORD *)(v3 + 224);
  v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
  v27 = 1;
  v25[0] = "Header: augmentation string";
  v26 = 3;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v22, v25, 1);
    v22 = *(_QWORD *)(v3 + 224);
  }
  return (*(__int64 (__fastcall **)(__int64, unsigned __int16 *, _QWORD))(*(_QWORD *)v22 + 512LL))(
           v22,
           a1 + 16,
           *((unsigned int *)a1 + 7));
}
