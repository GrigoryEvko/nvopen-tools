// Function: sub_E75E60
// Address: 0xe75e60
//
__int64 __fastcall sub_E75E60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r13
  char v13; // al
  __int64 v14; // rdx
  void (*v15)(); // rcx
  void (__fastcall *v16)(_QWORD *, __int64, __int64, __int64); // r8
  __int64 v17; // rcx
  __int64 v18; // rax
  void (*v19)(); // rcx
  __int64 v20; // rax
  void (*v21)(); // rcx
  __int64 v22; // rax
  void (*v23)(); // rcx
  __int64 v25; // rax
  void (*v26)(); // rcx
  __int64 v27; // rdx
  void (*v28)(); // rax
  char v29; // [rsp+Fh] [rbp-61h]
  __int64 v30[4]; // [rsp+10h] [rbp-60h] BYREF
  char v31; // [rsp+30h] [rbp-40h]
  char v32; // [rsp+31h] [rbp-3Fh]

  v6 = a1[1];
  v32 = 1;
  v30[0] = (__int64)"debug_list_header_start";
  v31 = 3;
  v7 = sub_E6C380(v6, v30, 1, a4, a5);
  v8 = a1[1];
  v32 = 1;
  v9 = v7;
  v31 = 3;
  v30[0] = (__int64)"debug_list_header_end";
  v12 = sub_E6C380(v8, v30, 1, v10, v11);
  v13 = *(_BYTE *)(a1[1] + 1906LL);
  if ( v13 != 1 )
  {
    v14 = *a1;
    v15 = *(void (**)())(*a1 + 120LL);
    v30[0] = (__int64)"Length";
    v32 = 1;
    v31 = 3;
    if ( v15 == nullsub_98 )
    {
      v16 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(v14 + 832);
      if ( !v13 )
      {
LABEL_4:
        v17 = 4;
        goto LABEL_5;
      }
    }
    else
    {
      v29 = v13;
      ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v15)(a1, v30, 1);
      v16 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a1 + 832LL);
      if ( !v29 )
        goto LABEL_4;
    }
    BUG();
  }
  v25 = *a1;
  v26 = *(void (**)())(*a1 + 120LL);
  v30[0] = (__int64)"DWARF64 mark";
  v32 = 1;
  v31 = 3;
  if ( v26 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v26)(a1, v30, 1);
    v25 = *a1;
  }
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(v25 + 536))(a1, 0xFFFFFFFFLL, 4);
  v27 = *a1;
  v28 = *(void (**)())(*a1 + 120LL);
  v32 = 1;
  v30[0] = (__int64)"Length";
  if ( v28 == nullsub_98 )
  {
    v16 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(v27 + 832);
  }
  else
  {
    v31 = 3;
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v28)(a1, v30, 1);
    v16 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a1 + 832LL);
  }
  v17 = 8;
LABEL_5:
  v16(a1, v12, v9, v17);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 208LL))(a1, v9, 0);
  v18 = *a1;
  v19 = *(void (**)())(*a1 + 120LL);
  v32 = 1;
  v30[0] = (__int64)"Version";
  v31 = 3;
  if ( v19 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v19)(a1, v30, 1);
    v18 = *a1;
  }
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(v18 + 536))(a1, *(unsigned __int16 *)(a1[1] + 1904LL), 2);
  v20 = *a1;
  v21 = *(void (**)())(*a1 + 120LL);
  v32 = 1;
  v30[0] = (__int64)"Address size";
  v31 = 3;
  if ( v21 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v21)(a1, v30, 1);
    v20 = *a1;
  }
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(v20 + 536))(
    a1,
    *(unsigned int *)(*(_QWORD *)(a1[1] + 152LL) + 8LL),
    1);
  v22 = *a1;
  v23 = *(void (**)())(*a1 + 120LL);
  v32 = 1;
  v30[0] = (__int64)"Segment selector size";
  v31 = 3;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v23)(a1, v30, 1);
    v22 = *a1;
  }
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(v22 + 536))(a1, 0, 1);
  return v12;
}
