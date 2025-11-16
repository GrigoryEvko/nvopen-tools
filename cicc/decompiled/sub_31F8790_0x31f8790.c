// Function: sub_31F8790
// Address: 0x31f8790
//
__int64 __fastcall sub_31F8790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r12
  void (*v19)(); // rax
  __int64 *v20; // rdi
  __int64 v21; // rax
  __int64 (*v22)(void); // rdx
  char v24; // al
  bool v25; // zf
  void (*v26)(); // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  const char *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-68h]
  _QWORD v33[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v34; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(v6 + 2480);
  v8 = v6 + 8;
  if ( !v7 )
    v7 = v8;
  v12 = sub_E6C430(v7, a2, a3, a4, a5);
  v13 = *(_QWORD *)(a1 + 16);
  v14 = *(_QWORD *)(v13 + 2480);
  v15 = v13 + 8;
  if ( !v14 )
    v14 = v15;
  v16 = sub_E6C430(v14, a2, v9, v10, v11);
  v17 = *(_QWORD *)(a1 + 528);
  v18 = v16;
  v19 = *(void (**)())(*(_QWORD *)v17 + 120LL);
  v33[0] = "Record length";
  v34 = 259;
  if ( v19 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v19)(v17, v33, 1);
    v17 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 832LL))(v17, v18, v12, 2);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 528) + 208LL))(*(_QWORD *)(a1 + 528), v12, 0);
  v20 = *(__int64 **)(a1 + 528);
  v21 = *v20;
  v22 = *(__int64 (**)(void))(*v20 + 96);
  if ( v22 != sub_C13EE0 )
  {
    v24 = v22();
    v20 = *(__int64 **)(a1 + 528);
    v25 = v24 == 0;
    v21 = *v20;
    if ( !v25 )
    {
      v32 = *(_QWORD *)(a1 + 528);
      v26 = *(void (**)())(v21 + 120);
      v27 = sub_37079F0();
      v29 = v27 + 40 * v28;
      if ( v27 == v29 )
      {
LABEL_18:
        v31 = 0;
        v30 = byte_3F871B3;
      }
      else
      {
        while ( (_WORD)a2 != *(_WORD *)(v27 + 32) )
        {
          v27 += 40;
          if ( v29 == v27 )
            goto LABEL_18;
        }
        v30 = *(const char **)v27;
        v31 = *(_QWORD *)(v27 + 8);
      }
      v33[2] = v30;
      v34 = 1283;
      v33[0] = "Record kind: ";
      v33[3] = v31;
      if ( v26 != nullsub_98 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v26)(v32, v33, 1);
      v20 = *(__int64 **)(a1 + 528);
      v21 = *v20;
    }
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(v21 + 536))(v20, (unsigned __int16)a2, 2);
  return v18;
}
