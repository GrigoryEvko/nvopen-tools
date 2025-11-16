// Function: sub_37299F0
// Address: 0x37299f0
//
unsigned __int64 __fastcall sub_37299F0(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v5; // rdx
  __int64 v6; // r8
  _QWORD *v7; // rsi
  __int64 (*v8)(); // rax
  _QWORD *v9; // rdi
  char v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 *i; // r15
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r10d
  void (*v17)(); // rcx
  unsigned __int64 result; // rax
  int *j; // r15
  __int64 (__fastcall *v20)(_QWORD *, _QWORD, _QWORD); // r8
  int v21; // ecx
  _QWORD *v22; // rdi
  __int64 v23; // r9
  void (*v24)(); // r8
  __int64 v25; // rdi
  void (*v26)(); // rax
  void (*v27)(void); // rax
  __int64 v28; // r8
  void (*v29)(); // rax
  void (*v30)(); // rax
  __int64 v31; // [rsp+10h] [rbp-80h]
  int v33; // [rsp+18h] [rbp-78h]
  __int64 *v34; // [rsp+28h] [rbp-68h]
  unsigned __int64 v35; // [rsp+28h] [rbp-68h]
  _QWORD v36[2]; // [rsp+30h] [rbp-60h] BYREF
  int v37; // [rsp+40h] [rbp-50h]
  __int16 v38; // [rsp+50h] [rbp-40h]

  v5 = *(_QWORD **)(a1 + 8);
  v6 = v5[28];
  v7 = (_QWORD *)v5[29];
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 96LL);
  if ( v8 == sub_C13EE0 )
  {
    v9 = *(_QWORD **)(a1 + 8);
    v10 = 0;
    v11 = (_QWORD *)v7[75];
    LODWORD(v12) = 0;
    v34 = (__int64 *)v7[74];
  }
  else
  {
    v10 = ((__int64 (__fastcall *)(__int64))v8)(v6);
    if ( v10 )
    {
      if ( v7[74] == v7[75] )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          a3,
          0);
        goto LABEL_27;
      }
      v5 = *(_QWORD **)(a1 + 8);
      v28 = v5[28];
      v9 = v5;
      v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
      v36[0] = ">> Catch TypeInfos <<";
      v38 = 259;
      if ( v29 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v29)(v28, v36, 1);
        v5 = *(_QWORD **)(a1 + 8);
        v28 = v5[28];
        v9 = v5;
      }
      v30 = *(void (**)())(*(_QWORD *)v28 + 160LL);
      if ( v30 != nullsub_99 )
      {
        ((void (__fastcall *)(__int64))v30)(v28);
        v5 = *(_QWORD **)(a1 + 8);
        v9 = v5;
      }
      v11 = (_QWORD *)v7[75];
      v34 = (__int64 *)v7[74];
      v12 = v11 - v34;
    }
    else
    {
      v5 = *(_QWORD **)(a1 + 8);
      LODWORD(v12) = 0;
      v11 = (_QWORD *)v7[75];
      v34 = (__int64 *)v7[74];
      v9 = v5;
    }
  }
  if ( v11 != v34 )
  {
    for ( i = v11 - 1; ; --i )
    {
      v14 = *i;
      if ( v10 )
      {
        v15 = v9[28];
        v16 = v12 - 1;
        v17 = *(void (**)())(*(_QWORD *)v15 + 120LL);
        v36[0] = "TypeInfo ";
        v37 = v12;
        v38 = 2563;
        if ( v17 != nullsub_98 )
        {
          v31 = v14;
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v15, v36, 1);
          v9 = *(_QWORD **)(a1 + 8);
          v16 = v12 - 1;
          v14 = v31;
        }
        LODWORD(v12) = v16;
      }
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*v9 + 440LL))(v9, v14, a2);
      if ( i == v34 )
        break;
      v9 = *(_QWORD **)(a1 + 8);
    }
    v5 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)v5[28] + 208LL))(v5[28], a3, 0);
  if ( !v10 )
  {
    result = v7[77];
    v35 = v7[78];
    goto LABEL_14;
  }
LABEL_27:
  result = (unsigned __int64)v7;
  if ( v7[78] == v7[77] )
    return result;
  v25 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v26 = *(void (**)())(*(_QWORD *)v25 + 120LL);
  v36[0] = ">> Filter TypeInfos <<";
  v38 = 259;
  if ( v26 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v26)(v25, v36, 1);
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  }
  v27 = *(void (**)(void))(*(_QWORD *)v25 + 160LL);
  if ( v27 != nullsub_99 )
    v27();
  v10 = 1;
  LODWORD(v12) = 0;
  result = v7[77];
  v35 = v7[78];
LABEL_14:
  if ( result < v35 )
  {
    for ( j = (int *)result; v35 > (unsigned __int64)j; ++j )
    {
      while ( 1 )
      {
        v21 = *j;
        v22 = *(_QWORD **)(a1 + 8);
        if ( !v10 )
          break;
        LODWORD(v12) = v12 - 1;
        if ( v21 )
        {
          v23 = v22[28];
          v24 = *(void (**)())(*(_QWORD *)v23 + 120LL);
          v36[0] = "FilterInfo ";
          v37 = v12;
          v38 = 2563;
          if ( v24 != nullsub_98 )
          {
            v33 = v21;
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v24)(v23, v36, 1);
            v22 = *(_QWORD **)(a1 + 8);
            v21 = v33;
          }
          v20 = *(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*v22 + 440LL);
          goto LABEL_21;
        }
        v20 = *(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*v22 + 440LL);
LABEL_18:
        ++j;
        result = v20(v22, 0, a2);
        if ( v35 <= (unsigned __int64)j )
          return result;
      }
      v20 = *(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*v22 + 440LL);
      if ( !v21 )
        goto LABEL_18;
LABEL_21:
      result = v20(v22, *(_QWORD *)(v7[74] + 8LL * (unsigned int)(v21 - 1)), a2);
    }
  }
  return result;
}
