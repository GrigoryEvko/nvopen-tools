// Function: sub_39A9C00
// Address: 0x39a9c00
//
void __fastcall sub_39A9C00(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v4; // rdx
  __int64 v5; // r8
  _QWORD *v6; // rcx
  __int64 (*v7)(); // rax
  __int64 *v8; // rax
  _QWORD *v9; // rdi
  char v10; // r13
  __int64 v11; // r12
  __int64 *i; // r15
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r10d
  void (*v16)(); // rcx
  unsigned __int64 v17; // r14
  int *v18; // rcx
  int *v19; // r15
  int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // r9
  void (*v23)(); // r8
  __int64 v24; // rdi
  void (*v25)(); // rax
  void (*v26)(void); // rax
  __int64 v27; // r8
  void (*v28)(); // rax
  void (*v29)(); // rax
  __int64 v30; // [rsp+10h] [rbp-90h]
  _QWORD *v32; // [rsp+20h] [rbp-80h]
  __int64 *v33; // [rsp+28h] [rbp-78h]
  int v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+30h] [rbp-70h]
  char *v36; // [rsp+50h] [rbp-50h] BYREF
  __int64 v37; // [rsp+58h] [rbp-48h]
  __int16 v38; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD **)(a1 + 8);
  v5 = v4[32];
  v6 = (_QWORD *)v4[33];
  v32 = v6;
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 80LL);
  if ( v7 == sub_168DB50 )
  {
    v8 = (__int64 *)v6[67];
    v9 = *(_QWORD **)(a1 + 8);
    v10 = 0;
    LODWORD(v11) = 0;
    v33 = (__int64 *)v6[66];
  }
  else
  {
    v10 = ((__int64 (__fastcall *)(__int64))v7)(v5);
    if ( v10 )
    {
      if ( v32[66] == v32[67] )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
          a3,
          0);
        goto LABEL_22;
      }
      v4 = *(_QWORD **)(a1 + 8);
      v27 = v4[32];
      v9 = v4;
      v28 = *(void (**)())(*(_QWORD *)v27 + 104LL);
      v36 = ">> Catch TypeInfos <<";
      v38 = 259;
      if ( v28 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, char **, __int64))v28)(v27, &v36, 1);
        v4 = *(_QWORD **)(a1 + 8);
        v27 = v4[32];
        v9 = v4;
      }
      v29 = *(void (**)())(*(_QWORD *)v27 + 144LL);
      if ( v29 != nullsub_581 )
      {
        ((void (__fastcall *)(__int64))v29)(v27);
        v4 = *(_QWORD **)(a1 + 8);
        v9 = v4;
      }
      v8 = (__int64 *)v32[67];
      v33 = (__int64 *)v32[66];
      v11 = v8 - v33;
    }
    else
    {
      v4 = *(_QWORD **)(a1 + 8);
      LODWORD(v11) = 0;
      v8 = (__int64 *)v32[67];
      v9 = v4;
      v33 = (__int64 *)v32[66];
    }
  }
  if ( v33 != v8 )
  {
    for ( i = v8 - 1; ; --i )
    {
      v13 = *i;
      if ( v10 )
      {
        v14 = v9[32];
        LODWORD(v35) = v11;
        v15 = v11 - 1;
        v16 = *(void (**)())(*(_QWORD *)v14 + 104LL);
        v36 = "TypeInfo ";
        v38 = 2563;
        v37 = v35;
        if ( v16 != nullsub_580 )
        {
          v30 = v13;
          ((void (__fastcall *)(__int64, char **, __int64))v16)(v14, &v36, 1);
          v9 = *(_QWORD **)(a1 + 8);
          v15 = v11 - 1;
          v13 = v30;
        }
        LODWORD(v11) = v15;
      }
      sub_397C360(v9, v13, a2);
      if ( i == v33 )
        break;
      v9 = *(_QWORD **)(a1 + 8);
    }
    v4 = *(_QWORD **)(a1 + 8);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)v4[32] + 176LL))(v4[32], a3, 0);
  if ( !v10 )
  {
    v17 = v32[70];
    v18 = (int *)v32[69];
    goto LABEL_14;
  }
LABEL_22:
  if ( v32[70] == v32[69] )
    return;
  v24 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v25 = *(void (**)())(*(_QWORD *)v24 + 104LL);
  v36 = ">> Filter TypeInfos <<";
  v38 = 259;
  if ( v25 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v25)(v24, &v36, 1);
    v24 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  }
  v26 = *(void (**)(void))(*(_QWORD *)v24 + 144LL);
  if ( v26 != nullsub_581 )
    v26();
  v10 = 1;
  LODWORD(v11) = 0;
  v17 = v32[70];
  v18 = (int *)v32[69];
LABEL_14:
  if ( (unsigned __int64)v18 < v17 )
  {
    v19 = v18;
    do
    {
      v20 = *v19;
      v21 = *(_QWORD *)(a1 + 8);
      if ( v10 )
      {
        LODWORD(v11) = v11 - 1;
        if ( v20 < 0 )
        {
          v22 = *(_QWORD *)(v21 + 256);
          LODWORD(v35) = v11;
          v23 = *(void (**)())(*(_QWORD *)v22 + 104LL);
          v36 = "FilterInfo ";
          v37 = v35;
          v38 = 2563;
          if ( v23 != nullsub_580 )
          {
            v34 = v20;
            ((void (__fastcall *)(__int64, char **, __int64))v23)(v22, &v36, 1);
            v21 = *(_QWORD *)(a1 + 8);
            v20 = v34;
          }
        }
      }
      ++v19;
      sub_397C0C0(v21, (unsigned int)v20, 0);
    }
    while ( v17 > (unsigned __int64)v19 );
  }
}
