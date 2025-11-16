// Function: sub_32591C0
// Address: 0x32591c0
//
__int64 (*__fastcall sub_32591C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5))()
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 (*result)(); // rax
  void (*v9)(); // rax
  unsigned __int8 *v10; // rax
  void (*v11)(); // rax
  unsigned __int8 *v12; // rax
  void (*v13)(); // rax
  void (*v14)(); // rax
  int *v15; // r12
  unsigned __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned __int8 *v22; // r14
  __int64 v23; // rax
  unsigned __int8 *v24; // r13
  unsigned __int8 *v25; // rax
  unsigned __int8 *v26; // rax
  __int64 v27; // rax
  bool v28; // zf
  _QWORD *v29; // [rsp+8h] [rbp-88h]
  char v33; // [rsp+2Fh] [rbp-61h]
  _QWORD v34[4]; // [rsp+30h] [rbp-60h] BYREF
  char v35; // [rsp+50h] [rbp-40h]
  char v36; // [rsp+51h] [rbp-3Fh]

  v5 = a5;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 224);
  v33 = 0;
  v29 = *(_QWORD **)(v6 + 216);
  result = *(__int64 (**)())(*(_QWORD *)v7 + 96LL);
  if ( result != sub_C13EE0 )
  {
    result = (__int64 (*)())((__int64 (__fastcall *)(__int64))result)(v7);
    v33 = (char)result;
  }
  while ( (_DWORD)v5 != -1 )
  {
    v15 = (int *)(*(_QWORD *)(a2 + 512) + 24 * v5);
    v16 = *((_QWORD *)v15 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *((_BYTE *)v15 + 4) )
    {
      v27 = sub_3258B50(*((_QWORD *)v15 + 2) & 0xFFFFFFFFFFFFFFF8LL);
      v22 = (unsigned __int8 *)sub_3258F50(a1, v27);
      v24 = (unsigned __int8 *)sub_E81A90(0, v29, 0, 0);
    }
    else
    {
      v17 = *((_QWORD *)v15 + 1);
      if ( v17 )
      {
        v18 = sub_31DB510(*(_QWORD *)(a1 + 8), v17);
        v22 = (unsigned __int8 *)sub_3258F50(a1, v18);
      }
      else
      {
        v18 = (__int64)v29;
        v22 = (unsigned __int8 *)sub_E81A90(1, v29, 0, 0);
      }
      v23 = sub_2E309C0(v16, v18, v19, v20, v21);
      v24 = (unsigned __int8 *)sub_3258F50(a1, v23);
    }
    v36 = 1;
    v34[0] = "LabelStart";
    v35 = 3;
    if ( !v33 )
    {
      v25 = (unsigned __int8 *)sub_E808D0(a3, 0x72u, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0);
      sub_E9A5B0(v7, v25);
      v26 = (unsigned __int8 *)sub_3258F90(a1, a4);
      sub_E9A5B0(v7, v26);
      sub_E9A5B0(v7, v22);
      goto LABEL_11;
    }
    v9 = *(void (**)())(*(_QWORD *)v7 + 120LL);
    if ( v9 != nullsub_98 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v9)(v7, v34, 1);
    v10 = (unsigned __int8 *)sub_E808D0(a3, 0x72u, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0);
    sub_E9A5B0(v7, v10);
    v36 = 1;
    v34[0] = "LabelEnd";
    v35 = 3;
    v11 = *(void (**)())(*(_QWORD *)v7 + 120LL);
    if ( v11 != nullsub_98 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v11)(v7, v34, 1);
    v12 = (unsigned __int8 *)sub_3258F90(a1, a4);
    sub_E9A5B0(v7, v12);
    if ( *((_BYTE *)v15 + 4) )
    {
      v36 = 1;
      v34[0] = "FinallyFunclet";
      v35 = 3;
      v13 = *(void (**)())(*(_QWORD *)v7 + 120LL);
      if ( v13 == nullsub_98 )
        goto LABEL_9;
LABEL_25:
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v13)(v7, v34, 1);
      goto LABEL_9;
    }
    v28 = *((_QWORD *)v15 + 1) == 0;
    v36 = 1;
    if ( v28 )
      v34[0] = "CatchAll";
    else
      v34[0] = "FilterFunction";
    v35 = 3;
    v13 = *(void (**)())(*(_QWORD *)v7 + 120LL);
    if ( v13 != nullsub_98 )
      goto LABEL_25;
LABEL_9:
    sub_E9A5B0(v7, v22);
    if ( *((_BYTE *)v15 + 4) )
    {
      v36 = 1;
      v34[0] = "Null";
      v35 = 3;
      v14 = *(void (**)())(*(_QWORD *)v7 + 120LL);
      if ( v14 == nullsub_98 )
        goto LABEL_11;
    }
    else
    {
      v36 = 1;
      v34[0] = "ExceptionHandler";
      v35 = 3;
      v14 = *(void (**)())(*(_QWORD *)v7 + 120LL);
      if ( v14 == nullsub_98 )
        goto LABEL_11;
    }
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v14)(v7, v34, 1);
LABEL_11:
    result = (__int64 (*)())sub_E9A5B0(v7, v24);
    v5 = *v15;
  }
  return result;
}
