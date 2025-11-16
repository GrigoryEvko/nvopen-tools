// Function: sub_39ACD80
// Address: 0x39acd80
//
__int64 (*__fastcall sub_39ACD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5))()
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 (*result)(); // rax
  void (*v9)(); // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  void (*v12)(); // rax
  int *v13; // r12
  unsigned __int64 v14; // r14
  __int64 v15; // rsi
  __int64 v16; // rax
  unsigned int *v17; // r15
  __int64 v18; // rax
  unsigned int *v19; // r14
  void (*v20)(); // rax
  unsigned __int64 v21; // rax
  unsigned int *v22; // rax
  void (*v23)(); // rax
  unsigned __int64 v24; // rax
  unsigned int *v25; // rax
  const char *v26; // rax
  unsigned __int64 v27; // rax
  unsigned int *v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-90h]
  __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+28h] [rbp-68h]
  unsigned __int64 v37; // [rsp+30h] [rbp-60h]
  unsigned __int64 v38; // [rsp+30h] [rbp-60h]
  unsigned __int64 v39; // [rsp+30h] [rbp-60h]
  char v40; // [rsp+3Fh] [rbp-51h]
  _QWORD v41[2]; // [rsp+40h] [rbp-50h] BYREF
  char v42; // [rsp+50h] [rbp-40h]
  char v43; // [rsp+51h] [rbp-3Fh]

  v5 = a5;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 256);
  v40 = 0;
  v30 = *(_QWORD *)(v6 + 248);
  result = *(__int64 (**)())(*(_QWORD *)v7 + 80LL);
  if ( result != sub_168DB50 )
  {
    result = (__int64 (*)())((__int64 (__fastcall *)(__int64))result)(v7);
    v40 = (char)result;
  }
  while ( (_DWORD)v5 != -1 )
  {
    v13 = (int *)(*(_QWORD *)(a2 + 480) + 24 * v5);
    v14 = *((_QWORD *)v13 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *((_BYTE *)v13 + 4) )
    {
      v29 = sub_39AC850(*((_QWORD *)v13 + 2) & 0xFFFFFFFFFFFFFFF8LL);
      v17 = (unsigned int *)sub_39ACBF0(a1, v29);
      v19 = (unsigned int *)sub_38CB470(0, v30);
    }
    else
    {
      v15 = *((_QWORD *)v13 + 1);
      if ( v15 )
      {
        v16 = sub_396EAF0(*(_QWORD *)(a1 + 8), v15);
        v17 = (unsigned int *)sub_39ACBF0(a1, v16);
      }
      else
      {
        v17 = (unsigned int *)sub_38CB470(1, v30);
      }
      v18 = sub_1DD5A70(v14);
      v19 = (unsigned int *)sub_39ACBF0(a1, v18);
    }
    v43 = 1;
    v41[0] = "LabelStart";
    v42 = 3;
    if ( v40 )
    {
      v20 = *(void (**)())(*(_QWORD *)v7 + 104LL);
      if ( v20 != nullsub_580 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v20)(v7, v41, 1);
      v34 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
      v37 = sub_38CB470(1, v34);
      v21 = sub_39ACBF0(a1, a3);
      v22 = (unsigned int *)sub_38CB1F0(0, v21, v37, v34, 0);
      sub_38DDD30(v7, v22);
      v43 = 1;
      v41[0] = "LabelEnd";
      v42 = 3;
      v23 = *(void (**)())(*(_QWORD *)v7 + 104LL);
      if ( v23 != nullsub_580 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v7, v41, 1);
    }
    else
    {
      v36 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
      v39 = sub_38CB470(1, v36);
      v27 = sub_39ACBF0(a1, a3);
      v28 = (unsigned int *)sub_38CB1F0(0, v27, v39, v36, 0);
      sub_38DDD30(v7, v28);
    }
    v35 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
    v38 = sub_38CB470(1, v35);
    v24 = sub_39ACBF0(a1, a4);
    v25 = (unsigned int *)sub_38CB1F0(0, v24, v38, v35, 0);
    sub_38DDD30(v7, v25);
    v26 = "FinallyFunclet";
    if ( !*((_BYTE *)v13 + 4) )
    {
      v26 = "CatchAll";
      if ( *((_QWORD *)v13 + 1) )
        v26 = "FilterFunction";
    }
    v43 = 1;
    v41[0] = v26;
    v42 = 3;
    if ( !v40 )
    {
      sub_38DDD30(v7, v17);
      goto LABEL_7;
    }
    v9 = *(void (**)())(*(_QWORD *)v7 + 104LL);
    if ( v9 != nullsub_580 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v9)(v7, v41, 1);
    sub_38DDD30(v7, v17);
    if ( *((_BYTE *)v13 + 4) )
    {
      v43 = 1;
      v41[0] = "Null";
      v42 = 3;
      v12 = *(void (**)())(*(_QWORD *)v7 + 104LL);
      if ( v12 == nullsub_580 )
        goto LABEL_7;
    }
    else
    {
      v43 = 1;
      v41[0] = "ExceptionHandler";
      v42 = 3;
      v12 = *(void (**)())(*(_QWORD *)v7 + 104LL);
      if ( v12 == nullsub_580 )
        goto LABEL_7;
    }
    ((void (__fastcall *)(__int64, _QWORD *, __int64, __int64, __int64, void (*)()))v12)(
      v7,
      v41,
      1,
      v10,
      v11,
      nullsub_580);
LABEL_7:
    result = (__int64 (*)())sub_38DDD30(v7, v19);
    v5 = *v13;
  }
  return result;
}
