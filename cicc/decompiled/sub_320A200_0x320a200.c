// Function: sub_320A200
// Address: 0x320a200
//
__int64 __fastcall sub_320A200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  void (*v11)(); // rax
  __int64 v12; // rdi
  void (*v13)(); // rax
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // rdi
  void (*v17)(); // rax
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rdi
  void (*v23)(); // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 *v28; // rsi
  __int64 *v29; // rbx
  __int64 *v30; // r13
  __int64 v31; // rsi
  _QWORD v33[4]; // [rsp+0h] [rbp-60h] BYREF
  char v34; // [rsp+20h] [rbp-40h]
  char v35; // [rsp+21h] [rbp-3Fh]

  v8 = sub_31F8790(a1, 4355, a3, a4, a5);
  v9 = *(_QWORD *)(a1 + 528);
  v10 = v8;
  v11 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v35 = 1;
  v33[0] = "PtrParent";
  v34 = 3;
  if ( v11 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v11)(v9, v33, 1);
    v9 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v9 + 536LL))(v9, 0, 4);
  v12 = *(_QWORD *)(a1 + 528);
  v13 = *(void (**)())(*(_QWORD *)v12 + 120LL);
  v35 = 1;
  v33[0] = "PtrEnd";
  v34 = 3;
  if ( v13 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v13)(v12, v33, 1);
    v12 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v12 + 536LL))(v12, 0, 4);
  v14 = *(_QWORD *)(a1 + 528);
  v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
  v35 = 1;
  v33[0] = "Code size";
  v34 = 3;
  if ( v15 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v14, v33, 1);
    v14 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v14 + 832LL))(
    v14,
    *(_QWORD *)(a2 + 168),
    *(_QWORD *)(a2 + 160),
    4);
  v16 = *(_QWORD *)(a1 + 528);
  v17 = *(void (**)())(*(_QWORD *)v16 + 120LL);
  v35 = 1;
  v33[0] = "Function section relative address";
  v34 = 3;
  if ( v17 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v16, v33, 1);
    v16 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v16 + 368LL))(v16, *(_QWORD *)(a2 + 160), 0);
  v18 = *(_QWORD *)(a1 + 528);
  v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
  v35 = 1;
  v33[0] = "Function section index";
  v34 = 3;
  if ( v19 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v19)(v18, v33, 1);
    v18 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v18 + 360LL))(v18, *(_QWORD *)(a3 + 440));
  v22 = *(__int64 **)(a1 + 528);
  v23 = *(void (**)())(*v22 + 120);
  v35 = 1;
  v33[0] = "Lexical block name";
  v34 = 3;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v23)(v22, v33, 1);
    v22 = *(__int64 **)(a1 + 528);
  }
  sub_31F4F00(v22, *(const void **)(a2 + 176), *(_QWORD *)(a2 + 184), 3840, v20, v21);
  sub_31F8930(a1, v10);
  sub_3209920(a1, a3, *(_QWORD *)a2, *(unsigned int *)(a2 + 8), v24, v25);
  v26 = *(__int64 **)(a2 + 104);
  v27 = &v26[2 * *(unsigned int *)(a2 + 112)];
  while ( v27 != v26 )
  {
    v28 = v26;
    v26 += 2;
    sub_3208CF0(a1, v28);
  }
  v29 = *(__int64 **)(a2 + 136);
  v30 = &v29[*(unsigned int *)(a2 + 144)];
  while ( v30 != v29 )
  {
    v31 = *v29++;
    sub_320A200(a1, v31, a3);
  }
  return sub_31F93A0(a1, 6u);
}
