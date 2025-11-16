// Function: sub_31F94F0
// Address: 0x31f94f0
//
void __fastcall sub_31F94F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const char *v6; // rax
  size_t v7; // rdx
  __int64 v8; // r8
  size_t v9; // r14
  const char *v10; // rbx
  const void **v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdi
  void (*v14)(); // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  void (*v21)(); // rax
  __int64 v22; // rdi
  void (*v23)(); // rax
  __int64 v24; // rdi
  void (*v25)(); // rax
  __int64 v26; // rdi
  void (*v27)(); // rax
  __int64 v28; // rdi
  void (*v29)(); // rax
  __int64 v30; // rdi
  void (*v31)(); // rax
  __int64 v32; // rdi
  void (*v33)(); // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 *v36; // rdi
  void (*v37)(); // rax
  _QWORD *v38; // rdi
  __int64 v39; // [rsp+0h] [rbp-90h]
  _BYTE *v41; // [rsp+10h] [rbp-80h] BYREF
  size_t v42; // [rsp+18h] [rbp-78h]
  _QWORD v43[2]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v44[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v45; // [rsp+50h] [rbp-40h]

  v6 = sub_BD5D20(a2);
  v9 = v7;
  if ( !v7 )
  {
    v11 = (const void **)&v41;
    v41 = v43;
    goto LABEL_8;
  }
  v10 = v6;
  if ( *v6 == 1 )
  {
    v9 = v7 - 1;
    v10 = v6 + 1;
  }
  v41 = v43;
  v44[0] = v9;
  if ( v9 > 0xF )
  {
    v41 = (_BYTE *)sub_22409D0((__int64)&v41, v44, 0);
    v38 = v41;
    v43[0] = v44[0];
LABEL_30:
    memcpy(v38, v10, v9);
    v9 = v44[0];
    v11 = (const void **)&v41;
    goto LABEL_8;
  }
  if ( v9 == 1 )
  {
    v11 = (const void **)&v41;
    LOBYTE(v43[0]) = *v10;
    goto LABEL_8;
  }
  if ( v9 )
  {
    v38 = v43;
    goto LABEL_30;
  }
  v11 = (const void **)&v41;
LABEL_8:
  v42 = v9;
  v12 = 1027;
  v41[v9] = 0;
  v13 = *(_QWORD *)(a1 + 528);
  v14 = *(void (**)())(*(_QWORD *)v13 + 120LL);
  v44[0] = (unsigned __int64)"Symbol subsection for ";
  v44[2] = (unsigned __int64)&v41;
  v45 = 1027;
  if ( v14 != nullsub_98 )
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v14)(v13, v44, 1);
  v39 = sub_31F8650(a1, 241, v12, (__int64)v11, v8);
  v18 = sub_31F8790(a1, 4354, v15, v16, v17);
  v19 = *(_QWORD *)(a1 + 528);
  v20 = v18;
  v21 = *(void (**)())(*(_QWORD *)v19 + 120LL);
  v44[0] = (unsigned __int64)"PtrParent";
  v45 = 259;
  if ( v21 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v21)(v19, v44, 1);
    v19 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v19 + 536LL))(v19, 0, 4);
  v22 = *(_QWORD *)(a1 + 528);
  v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
  v44[0] = (unsigned __int64)"PtrEnd";
  v45 = 259;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v23)(v22, v44, 1);
    v22 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v22 + 536LL))(v22, 0, 4);
  v24 = *(_QWORD *)(a1 + 528);
  v25 = *(void (**)())(*(_QWORD *)v24 + 120LL);
  v44[0] = (unsigned __int64)"PtrNext";
  v45 = 259;
  if ( v25 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v25)(v24, v44, 1);
    v24 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v24 + 536LL))(v24, 0, 4);
  v26 = *(_QWORD *)(a1 + 528);
  v27 = *(void (**)())(*(_QWORD *)v26 + 120LL);
  v44[0] = (unsigned __int64)"Thunk section relative address";
  v45 = 259;
  if ( v27 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v27)(v26, v44, 1);
    v26 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v26 + 368LL))(v26, a4, 0);
  v28 = *(_QWORD *)(a1 + 528);
  v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
  v44[0] = (unsigned __int64)"Thunk section index";
  v45 = 259;
  if ( v29 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v29)(v28, v44, 1);
    v28 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v28 + 360LL))(v28, a4);
  v30 = *(_QWORD *)(a1 + 528);
  v31 = *(void (**)())(*(_QWORD *)v30 + 120LL);
  v44[0] = (unsigned __int64)"Code size";
  v45 = 259;
  if ( v31 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v31)(v30, v44, 1);
    v30 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v30 + 832LL))(
    v30,
    *(_QWORD *)(a3 + 448),
    a4,
    2);
  v32 = *(_QWORD *)(a1 + 528);
  v33 = *(void (**)())(*(_QWORD *)v32 + 120LL);
  v44[0] = (unsigned __int64)"Ordinal";
  v45 = 259;
  if ( v33 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v33)(v32, v44, 1);
    v32 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v32 + 536LL))(v32, 0, 1);
  v36 = *(__int64 **)(a1 + 528);
  v37 = *(void (**)())(*v36 + 120);
  v44[0] = (unsigned __int64)"Function name";
  v45 = 259;
  if ( v37 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, unsigned __int64 *, __int64))v37)(v36, v44, 1);
    v36 = *(__int64 **)(a1 + 528);
  }
  sub_31F4F00(v36, v41, v42, 3840, v34, v35);
  sub_31F8930(a1, v20);
  sub_31F93A0(a1, 0x114Fu);
  sub_31F8740(a1, v39);
  if ( v41 != (_BYTE *)v43 )
    j_j___libc_free_0((unsigned __int64)v41);
}
