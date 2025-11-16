// Function: sub_31F9B40
// Address: 0x31f9b40
//
__int64 __fastcall sub_31F9B40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 result; // rax
  __int16 *v5; // r15
  __int64 v7; // rdi
  __int64 v8; // r8
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
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rax
  void (*v25)(); // rax
  __int64 v26; // rdi
  void (*v27)(); // rax
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+20h] [rbp-80h]
  __int64 v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+30h] [rbp-70h]
  unsigned __int16 v36; // [rsp+3Eh] [rbp-62h]
  _QWORD v37[4]; // [rsp+40h] [rbp-60h] BYREF
  char v38; // [rsp+60h] [rbp-40h]
  char v39; // [rsp+61h] [rbp-3Fh]

  v3 = *(_QWORD *)(a2 + 424);
  result = *(_QWORD *)(a2 + 416);
  v30 = v3;
  if ( result != v3 )
  {
    v5 = *(__int16 **)(a2 + 416);
    do
    {
      v22 = *((_QWORD *)v5 + 3);
      v23 = *((_QWORD *)v5 + 4);
      v36 = *v5;
      v33 = *((_QWORD *)v5 + 1);
      v31 = *((_QWORD *)v5 + 2);
      v34 = *((_QWORD *)v5 + 5);
      v24 = sub_31F8790(a1, 4441, a3, v3, v33);
      v28 = *(_QWORD *)(a1 + 528);
      v29 = v33;
      v35 = v24;
      v25 = *(void (**)())(*(_QWORD *)v28 + 120LL);
      v39 = 1;
      v37[0] = "Base offset";
      v38 = 3;
      if ( v33 )
      {
        if ( v25 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v28, v37, 1);
          v28 = *(_QWORD *)(a1 + 528);
          v29 = v33;
        }
        v32 = v29;
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v28 + 368LL))(v28, v29, v31);
        v7 = *(_QWORD *)(a1 + 528);
        v8 = v32;
        v9 = *(void (**)())(*(_QWORD *)v7 + 120LL);
        v39 = 1;
        v37[0] = "Base section index";
        v38 = 3;
        if ( v9 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64, const char *, __int64))v9)(
            v7,
            v37,
            1,
            "Base section index",
            v32);
          v7 = *(_QWORD *)(a1 + 528);
          v8 = v32;
        }
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 360LL))(v7, v8);
      }
      else
      {
        if ( v25 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v28, v37, 1);
          v28 = *(_QWORD *)(a1 + 528);
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v28 + 536LL))(v28, 0, 4);
        v26 = *(_QWORD *)(a1 + 528);
        v27 = *(void (**)())(*(_QWORD *)v26 + 120LL);
        v39 = 1;
        v37[0] = "Base section index";
        v38 = 3;
        if ( v27 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v27)(v26, v37, 1);
          v26 = *(_QWORD *)(a1 + 528);
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v26 + 536LL))(v26, 0, 2);
      }
      v10 = *(_QWORD *)(a1 + 528);
      v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
      v39 = 1;
      v37[0] = "Switch type";
      v38 = 3;
      if ( v11 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v11)(v10, v37, 1);
        v10 = *(_QWORD *)(a1 + 528);
      }
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v10 + 536LL))(v10, v36, 2);
      v12 = *(_QWORD *)(a1 + 528);
      v13 = *(void (**)())(*(_QWORD *)v12 + 120LL);
      v39 = 1;
      v37[0] = "Branch offset";
      v38 = 3;
      if ( v13 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v13)(v12, v37, 1);
        v12 = *(_QWORD *)(a1 + 528);
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v12 + 368LL))(v12, v22, 0);
      v14 = *(_QWORD *)(a1 + 528);
      v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
      v39 = 1;
      v37[0] = "Table offset";
      v38 = 3;
      if ( v15 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v14, v37, 1);
        v14 = *(_QWORD *)(a1 + 528);
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v14 + 368LL))(v14, v23, 0);
      v16 = *(_QWORD *)(a1 + 528);
      v17 = *(void (**)())(*(_QWORD *)v16 + 120LL);
      v39 = 1;
      v37[0] = "Branch section index";
      v38 = 3;
      if ( v17 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v16, v37, 1);
        v16 = *(_QWORD *)(a1 + 528);
      }
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 360LL))(v16, v22);
      v18 = *(_QWORD *)(a1 + 528);
      v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
      v39 = 1;
      v37[0] = "Table section index";
      v38 = 3;
      if ( v19 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v19)(v18, v37, 1);
        v18 = *(_QWORD *)(a1 + 528);
      }
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 360LL))(v18, v23);
      v20 = *(_QWORD *)(a1 + 528);
      v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
      v39 = 1;
      v37[0] = "Entries count";
      v38 = 3;
      if ( v21 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v37, 1);
        v20 = *(_QWORD *)(a1 + 528);
      }
      v5 += 24;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v20 + 536LL))(v20, v34, 4);
      result = sub_31F8930(a1, v35);
    }
    while ( (__int16 *)v30 != v5 );
  }
  return result;
}
