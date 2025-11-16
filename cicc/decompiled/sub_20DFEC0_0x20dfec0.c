// Function: sub_20DFEC0
// Address: 0x20dfec0
//
void __fastcall sub_20DFEC0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 (*v7)(); // rax
  __int64 (*v8)(); // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // [rsp-8h] [rbp-128h]
  __int64 v37; // [rsp+10h] [rbp-110h]
  __int64 v38; // [rsp+10h] [rbp-110h]
  __int64 v39; // [rsp+10h] [rbp-110h]
  __int64 v40; // [rsp+18h] [rbp-108h]
  __int64 v41; // [rsp+18h] [rbp-108h]
  __int64 v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+18h] [rbp-108h]
  int v44; // [rsp+24h] [rbp-FCh] BYREF
  __int64 v45; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v46; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v47; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v48; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v49; // [rsp+48h] [rbp-D8h]
  _BYTE v50[208]; // [rsp+50h] [rbp-D0h] BYREF

  v4 = *(_QWORD *)(a2 + 64);
  v45 = v4;
  if ( v4 )
    sub_1623A60((__int64)&v45, v4, 2);
  v5 = a1[59];
  v6 = *(_QWORD *)(a2 + 24);
  v49 = 0x400000000LL;
  v46 = 0;
  v47 = 0;
  v48 = v50;
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 264LL);
  if ( v7 == sub_1D820E0 )
  {
    v8 = *(__int64 (**)())(*(_QWORD *)v5 + 624LL);
    if ( v8 == sub_1D918B0 )
      goto LABEL_24;
  }
  else
  {
    ((void (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v7)(v5, v6, &v46, &v47, &v48, 0);
    v5 = a1[59];
    v8 = *(__int64 (**)())(*(_QWORD *)v5 + 624LL);
    if ( v8 == sub_1D918B0 )
    {
LABEL_18:
      if ( v47 )
      {
LABEL_19:
        v21 = sub_20DFA10((__int64)a1, v6);
        v22 = *(int *)(v21 + 48);
        v11 = (_QWORD *)v21;
        v23 = a1[59];
        v44 = 0;
        v41 = a1[29] + 8 * v22;
        (*(void (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, int *))(*(_QWORD *)v23 + 288LL))(
          v23,
          v21,
          v46,
          0,
          0,
          0,
          &v45,
          &v44);
        *(_DWORD *)(v41 + 4) += v44;
        sub_1DD9570(v6, v46, (__int64)v11);
        sub_1DD8FE0((__int64)v11, v46, -1);
        v24 = *(int *)(v6 + 48);
        v25 = a1[29];
        v26 = a1[59];
        v44 = 0;
        v42 = v25 + 8 * v24;
        (*(void (__fastcall **)(__int64, __int64, int *))(*(_QWORD *)v26 + 280LL))(v26, v6, &v44);
        *(_DWORD *)(v42 + 4) -= v44;
        v27 = *(int *)(v6 + 48);
        v28 = a1[29];
        v29 = a1[59];
        v44 = 0;
        v43 = v28 + 8 * v27;
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, _BYTE *, _QWORD, __int64 *, int *))(*(_QWORD *)v29 + 288LL))(
          v29,
          v6,
          v11,
          v47,
          v48,
          (unsigned int)v49,
          &v45,
          &v44);
        *(_DWORD *)(v43 + 4) += v44;
        sub_20DF590((__int64)a1, v6);
        goto LABEL_11;
      }
LABEL_24:
      v47 = *(_QWORD *)(v6 + 8);
      goto LABEL_19;
    }
  }
  if ( ((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v8)(v5, &v48) )
    goto LABEL_18;
  if ( !v47 )
    goto LABEL_20;
  if ( (unsigned __int8)sub_20DF6F0((__int64)a1, a2, v47) )
  {
    v30 = *(int *)(v6 + 48);
    v31 = a1[29];
    v44 = 0;
    v32 = v31 + 8 * v30;
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)a1[59] + 280LL))(a1[59], v6);
    *(_DWORD *)(v32 + 4) -= v44;
    v33 = *(int *)(v6 + 48);
    v34 = a1[29];
    v44 = 0;
    v35 = v34 + 8 * v33;
    (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *, int *))(*(_QWORD *)a1[59]
                                                                                                 + 288LL))(
      a1[59],
      v6,
      v47,
      v46,
      v48,
      (unsigned int)v49,
      &v45,
      &v44);
    *(_DWORD *)(v35 + 4) += v44;
    sub_20DF590((__int64)a1, v6);
    goto LABEL_12;
  }
  if ( v47 )
  {
    v9 = sub_20DFA10((__int64)a1, v6);
    v10 = *(int *)(v9 + 48);
    v11 = (_QWORD *)v9;
    v12 = a1[59];
    v44 = 0;
    v37 = a1[29] + 8 * v10;
    (*(void (__fastcall **)(__int64, unsigned __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, int *))(*(_QWORD *)v12 + 288LL))(
      v12,
      v9,
      v47,
      0,
      0,
      0,
      &v45,
      &v44);
    *(_DWORD *)(v37 + 4) += v44;
    sub_1DD9570(v6, v47, (__int64)v11);
    sub_1DD8FE0((__int64)v11, v47, -1);
  }
  else
  {
LABEL_20:
    v11 = 0;
  }
  v13 = *(_QWORD *)(v6 + 8);
  v14 = *(int *)(v6 + 48);
  v44 = 0;
  v40 = v13;
  v38 = a1[29] + 8 * v14;
  (*(void (__fastcall **)(_QWORD, __int64, int *))(*(_QWORD *)a1[59] + 280LL))(a1[59], v6, &v44);
  *(_DWORD *)(v38 + 4) -= v44;
  v15 = *(int *)(v6 + 48);
  v16 = a1[29];
  v17 = a1[59];
  v44 = 0;
  v39 = v16 + 8 * v15;
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, _BYTE *, _QWORD, __int64 *, int *))(*(_QWORD *)v17 + 288LL))(
    v17,
    v6,
    v40,
    v46,
    v48,
    (unsigned int)v49,
    &v45,
    &v44);
  *(_DWORD *)(v39 + 4) += v44;
  sub_20DF590((__int64)a1, v6);
  v18 = v36;
  if ( !v11 )
    goto LABEL_12;
LABEL_11:
  v19 = a1[58];
  v20 = *(__int64 (**)())(*(_QWORD *)v19 + 328LL);
  if ( v20 != sub_1F49C90 && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v20)(v19, a1[57], v18) )
    sub_1DC3250((__int64)(a1 + 48), v11);
LABEL_12:
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  if ( v45 )
    sub_161E7C0((__int64)&v45, v45);
}
