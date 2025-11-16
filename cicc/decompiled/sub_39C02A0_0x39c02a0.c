// Function: sub_39C02A0
// Address: 0x39c02a0
//
unsigned __int64 __fastcall sub_39C02A0(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v5; // rdx
  __int64 v6; // r8
  _QWORD *v7; // rcx
  __int64 (*v8)(); // rax
  _QWORD *v9; // rdi
  char v10; // r14
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 *i; // r15
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r10d
  void (*v17)(); // rcx
  unsigned __int64 result; // rax
  int *j; // r15
  __int64 v20; // rsi
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
  __int64 v31; // [rsp+10h] [rbp-90h]
  int v33; // [rsp+18h] [rbp-88h]
  _QWORD *v34; // [rsp+20h] [rbp-80h]
  __int64 *v35; // [rsp+28h] [rbp-78h]
  unsigned __int64 v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h]
  char *v38; // [rsp+50h] [rbp-50h] BYREF
  __int64 v39; // [rsp+58h] [rbp-48h]
  __int16 v40; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD **)(a1 + 8);
  v6 = v5[32];
  v7 = (_QWORD *)v5[33];
  v34 = v7;
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 80LL);
  if ( v8 == sub_168DB50 )
  {
    v9 = *(_QWORD **)(a1 + 8);
    v10 = 0;
    v11 = (__int64 *)v7[67];
    LODWORD(v12) = 0;
    v35 = (__int64 *)v7[66];
  }
  else
  {
    v10 = ((__int64 (__fastcall *)(__int64))v8)(v6);
    if ( v10 )
    {
      if ( v34[66] == v34[67] )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
          a3,
          0);
        goto LABEL_24;
      }
      v5 = *(_QWORD **)(a1 + 8);
      v28 = v5[32];
      v9 = v5;
      v29 = *(void (**)())(*(_QWORD *)v28 + 104LL);
      v38 = ">> Catch TypeInfos <<";
      v40 = 259;
      if ( v29 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, char **, __int64))v29)(v28, &v38, 1);
        v5 = *(_QWORD **)(a1 + 8);
        v28 = v5[32];
        v9 = v5;
      }
      v30 = *(void (**)())(*(_QWORD *)v28 + 144LL);
      if ( v30 != nullsub_581 )
      {
        ((void (__fastcall *)(__int64))v30)(v28);
        v5 = *(_QWORD **)(a1 + 8);
        v9 = v5;
      }
      v11 = (__int64 *)v34[67];
      v35 = (__int64 *)v34[66];
      v12 = v11 - v35;
    }
    else
    {
      v5 = *(_QWORD **)(a1 + 8);
      LODWORD(v12) = 0;
      v11 = (__int64 *)v34[67];
      v35 = (__int64 *)v34[66];
      v9 = v5;
    }
  }
  if ( v11 != v35 )
  {
    for ( i = v11 - 1; ; --i )
    {
      v14 = *i;
      if ( v10 )
      {
        v15 = v9[32];
        LODWORD(v37) = v12;
        v16 = v12 - 1;
        v17 = *(void (**)())(*(_QWORD *)v15 + 104LL);
        v38 = "TypeInfo ";
        v39 = v37;
        v40 = 2563;
        if ( v17 != nullsub_580 )
        {
          v31 = v14;
          ((void (__fastcall *)(__int64, char **, __int64))v17)(v15, &v38, 1);
          v9 = *(_QWORD **)(a1 + 8);
          v16 = v12 - 1;
          v14 = v31;
        }
        LODWORD(v12) = v16;
      }
      sub_397C360(v9, v14, a2);
      if ( i == v35 )
        break;
      v9 = *(_QWORD **)(a1 + 8);
    }
    v5 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)v5[32] + 176LL))(v5[32], a3, 0);
  if ( !v10 )
  {
    result = v34[69];
    v36 = v34[70];
    goto LABEL_14;
  }
LABEL_24:
  result = (unsigned __int64)v34;
  if ( v34[70] == v34[69] )
    return result;
  v25 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v26 = *(void (**)())(*(_QWORD *)v25 + 104LL);
  v38 = ">> Filter TypeInfos <<";
  v40 = 259;
  if ( v26 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v26)(v25, &v38, 1);
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  }
  v27 = *(void (**)(void))(*(_QWORD *)v25 + 144LL);
  if ( v27 != nullsub_581 )
    v27();
  v10 = 1;
  LODWORD(v12) = 0;
  result = v34[69];
  v36 = v34[70];
LABEL_14:
  if ( v36 > result )
  {
    for ( j = (int *)result; v36 > (unsigned __int64)j; ++j )
    {
      v21 = *j;
      v22 = *(_QWORD **)(a1 + 8);
      if ( v10 )
      {
        LODWORD(v12) = v12 - 1;
        v20 = 0;
        if ( v21 )
        {
          v23 = v22[32];
          LODWORD(v37) = v12;
          v24 = *(void (**)())(*(_QWORD *)v23 + 104LL);
          v38 = "FilterInfo ";
          v40 = 2563;
          v39 = v37;
          if ( v24 != nullsub_580 )
          {
            v33 = v21;
            ((void (__fastcall *)(__int64, char **, __int64))v24)(v23, &v38, 1);
            v22 = *(_QWORD **)(a1 + 8);
            v21 = v33;
          }
LABEL_20:
          v20 = *(_QWORD *)(v34[66] + 8LL * (unsigned int)(v21 - 1));
        }
      }
      else
      {
        v20 = 0;
        if ( v21 )
          goto LABEL_20;
      }
      result = sub_397C360(v22, v20, a2);
    }
  }
  return result;
}
