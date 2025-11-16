// Function: sub_3252720
// Address: 0x3252720
//
__int64 __fastcall sub_3252720(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v4; // rdx
  __int64 v5; // r8
  _QWORD *v6; // rcx
  __int64 (*v7)(); // rax
  _QWORD *v8; // rdi
  char v9; // r13
  __int64 v10; // r12
  __int64 *v11; // rax
  __int64 *i; // r15
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r10d
  void (*v16)(); // rcx
  __int64 result; // rax
  unsigned __int64 v18; // r14
  int *v19; // r8
  int *v20; // r15
  int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // r9
  void (*v24)(); // rcx
  __int64 v25; // rdi
  void (*v26)(); // rax
  void (*v27)(void); // rax
  __int64 v28; // r8
  void (*v29)(); // rax
  void (*v30)(); // rax
  __int64 v31; // [rsp+10h] [rbp-80h]
  _QWORD *v33; // [rsp+20h] [rbp-70h]
  __int64 *v34; // [rsp+28h] [rbp-68h]
  int v35; // [rsp+28h] [rbp-68h]
  _QWORD v36[2]; // [rsp+30h] [rbp-60h] BYREF
  int v37; // [rsp+40h] [rbp-50h]
  __int16 v38; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD **)(a1 + 8);
  v5 = v4[28];
  v6 = (_QWORD *)v4[29];
  v33 = v6;
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 96LL);
  if ( v7 == sub_C13EE0 )
  {
    v8 = *(_QWORD **)(a1 + 8);
    v9 = 0;
    LODWORD(v10) = 0;
    v11 = (__int64 *)v6[75];
    v34 = (__int64 *)v6[74];
  }
  else
  {
    v9 = ((__int64 (__fastcall *)(__int64))v7)(v5);
    if ( v9 )
    {
      if ( v33[74] == v33[75] )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          a3,
          0);
        goto LABEL_22;
      }
      v4 = *(_QWORD **)(a1 + 8);
      v28 = v4[28];
      v8 = v4;
      v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
      v36[0] = ">> Catch TypeInfos <<";
      v38 = 259;
      if ( v29 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v29)(v28, v36, 1);
        v4 = *(_QWORD **)(a1 + 8);
        v28 = v4[28];
        v8 = v4;
      }
      v30 = *(void (**)())(*(_QWORD *)v28 + 160LL);
      if ( v30 != nullsub_99 )
      {
        ((void (__fastcall *)(__int64))v30)(v28);
        v4 = *(_QWORD **)(a1 + 8);
        v8 = v4;
      }
      v11 = (__int64 *)v33[75];
      v34 = (__int64 *)v33[74];
      v10 = v11 - v34;
    }
    else
    {
      LODWORD(v10) = 0;
      v11 = (__int64 *)v33[75];
      v34 = (__int64 *)v33[74];
      v4 = *(_QWORD **)(a1 + 8);
      v8 = v4;
    }
  }
  if ( v34 != v11 )
  {
    for ( i = v11 - 1; ; --i )
    {
      v13 = *i;
      if ( v9 )
      {
        v14 = v8[28];
        v15 = v10 - 1;
        v16 = *(void (**)())(*(_QWORD *)v14 + 120LL);
        v36[0] = "TypeInfo ";
        v37 = v10;
        v38 = 2563;
        if ( v16 != nullsub_98 )
        {
          v31 = v13;
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v16)(v14, v36, 1);
          v8 = *(_QWORD **)(a1 + 8);
          v15 = v10 - 1;
          v13 = v31;
        }
        LODWORD(v10) = v15;
      }
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*v8 + 440LL))(v8, v13, a2);
      if ( i == v34 )
        break;
      v8 = *(_QWORD **)(a1 + 8);
    }
    v4 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)v4[28] + 208LL))(v4[28], a3, 0);
  if ( !v9 )
  {
    result = (__int64)v33;
    v18 = v33[78];
    v19 = (int *)v33[77];
    goto LABEL_14;
  }
LABEL_22:
  result = (__int64)v33;
  if ( v33[78] == v33[77] )
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
  result = (__int64)v33;
  v9 = 1;
  LODWORD(v10) = 0;
  v18 = v33[78];
  v19 = (int *)v33[77];
LABEL_14:
  if ( v18 > (unsigned __int64)v19 )
  {
    v20 = v19;
    do
    {
      v21 = *v20;
      v22 = *(_QWORD **)(a1 + 8);
      if ( v9 )
      {
        LODWORD(v10) = v10 - 1;
        if ( v21 < 0 )
        {
          v23 = v22[28];
          v24 = *(void (**)())(*(_QWORD *)v23 + 120LL);
          v37 = v10;
          v36[0] = "FilterInfo ";
          v38 = 2563;
          if ( v24 != nullsub_98 )
          {
            v35 = v21;
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v24)(v23, v36, 1);
            v22 = *(_QWORD **)(a1 + 8);
            v21 = v35;
          }
        }
      }
      ++v20;
      result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD))(*v22 + 424LL))(
                 v22,
                 (unsigned int)v21,
                 0,
                 0);
    }
    while ( v18 > (unsigned __int64)v20 );
  }
  return result;
}
