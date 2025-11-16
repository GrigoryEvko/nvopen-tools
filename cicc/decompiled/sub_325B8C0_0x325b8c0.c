// Function: sub_325B8C0
// Address: 0x325b8c0
//
__int64 __fastcall sub_325B8C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // rcx
  char v10; // r15
  __int64 (*v11)(); // rax
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // rax
  const char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 result; // rax
  __int64 v24; // r14
  int v25; // eax
  __int64 v26; // r8
  __int64 v27; // rax
  void (*v28)(); // r9
  void (*v29)(); // rax
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // rax
  unsigned __int8 *v33; // rsi
  bool v34; // zf
  const char *v35; // rax
  void (*v36)(); // rax
  unsigned __int64 v37; // rdi
  char v38; // al
  __int64 v39; // r13
  __int64 (*v40)(); // rax
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 (*v43)(); // rax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  void (*v47)(); // rcx
  __int64 v48; // rax
  void (*v49)(); // rcx
  __int64 v50; // rax
  void (*v51)(); // rcx
  __int64 v52; // rax
  void (*v53)(); // rcx
  __int64 v54; // [rsp+18h] [rbp-78h]
  __int64 v55; // [rsp+18h] [rbp-78h]
  __int64 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+20h] [rbp-70h]
  __int64 v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  int v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  int v64; // [rsp+28h] [rbp-68h]
  __int64 v65; // [rsp+28h] [rbp-68h]
  _QWORD v66[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v67; // [rsp+50h] [rbp-40h]

  v3 = *(_QWORD *)a2;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v5 = sub_BD5D20(*(_QWORD *)a2);
  v8 = (__int64)v5;
  v9 = v6;
  if ( v6 && *v5 == 1 )
  {
    v9 = v6 - 1;
    v8 = (__int64)(v5 + 1);
  }
  v10 = 0;
  v11 = *(__int64 (**)())(*(_QWORD *)v4 + 96LL);
  if ( v11 != sub_C13EE0 )
  {
    v56 = v8;
    v60 = v9;
    v38 = ((__int64 (__fastcall *)(__int64))v11)(v4);
    v8 = v56;
    v9 = v60;
    v10 = v38;
  }
  v54 = v8;
  v57 = v9;
  v12 = *(_QWORD *)(a2 + 88);
  sub_325B570(a1, v12, v8, v9, v8, v7);
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
  v66[0] = v54;
  v67 = 261;
  v66[1] = v57;
  v58 = sub_E6C840(v13, (__int64)v66, v14, v57);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v4 + 608LL))(v4, 2, 0, 1, 0);
  v15 = (unsigned __int8 *)v58;
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v4 + 208LL))(v4, v58, 0);
  v16 = (unsigned __int8 *)sub_B2E500(v3);
  v17 = sub_BD3990(v16, v58);
  v18 = sub_BD5D20((__int64)v17);
  if ( v19 == 16
    && (v19 = *(_QWORD *)v18 ^ 0x5F7470656378655FLL, (v20 = v19 | *((_QWORD *)v18 + 1) ^ 0x3472656C646E6168LL) == 0) )
  {
    v39 = *(_QWORD *)(a2 + 48);
    if ( *(_DWORD *)(v39 + 68) == -1 )
    {
      v42 = -2;
    }
    else
    {
      LODWORD(v66[0]) = 0;
      v40 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
      if ( v40 == sub_2DD19D0 )
        BUG();
      v41 = v40();
      v42 = (*(int (__fastcall **)(__int64, __int64, _QWORD, _QWORD *))(*(_QWORD *)v41 + 224LL))(
              v41,
              a2,
              *(unsigned int *)(v39 + 68),
              v66);
    }
    if ( *(_DWORD *)(v12 + 752) == 0x7FFFFFFF )
    {
      v65 = 9999;
    }
    else
    {
      LODWORD(v66[0]) = 0;
      v61 = v42;
      v43 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
      if ( v43 == sub_2DD19D0 )
        BUG();
      v44 = v43();
      v45 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD *))(*(_QWORD *)v44 + 224LL))(
              v44,
              a2,
              *(unsigned int *)(v12 + 752),
              v66);
      v42 = v61;
      v65 = v45;
    }
    v66[0] = "GSCookieOffset";
    v67 = 259;
    v46 = *(_QWORD *)v4;
    if ( v10 )
    {
      v47 = *(void (**)())(v46 + 120);
      if ( v47 != nullsub_98 )
      {
        v62 = v42;
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v47)(v4, v66, 1);
        v46 = *(_QWORD *)v4;
        v42 = v62;
      }
      (*(void (__fastcall **)(__int64, __int64, __int64))(v46 + 536))(v4, v42, 4);
      v66[0] = "GSCookieXOROffset";
      v67 = 259;
      v48 = *(_QWORD *)v4;
      v49 = *(void (**)())(*(_QWORD *)v4 + 120LL);
      if ( v49 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v49)(v4, v66, 1);
        v48 = *(_QWORD *)v4;
      }
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(v48 + 536))(v4, 0, 4);
      v66[0] = "EHCookieOffset";
      v67 = 259;
      v50 = *(_QWORD *)v4;
      v51 = *(void (**)())(*(_QWORD *)v4 + 120LL);
      if ( v51 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v51)(v4, v66, 1);
        v50 = *(_QWORD *)v4;
      }
      (*(void (__fastcall **)(__int64, __int64, __int64))(v50 + 536))(v4, v65, 4);
      v66[0] = "EHCookieXOROffset";
      v67 = 259;
      v52 = *(_QWORD *)v4;
      v53 = *(void (**)())(*(_QWORD *)v4 + 120LL);
      if ( v53 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v53)(v4, v66, 1);
        v52 = *(_QWORD *)v4;
      }
    }
    else
    {
      (*(void (__fastcall **)(__int64, __int64, __int64))(v46 + 536))(v4, v42, 4);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v4 + 536LL))(v4, 0, 4);
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v4 + 536LL))(v4, v65, 4);
      v52 = *(_QWORD *)v4;
    }
    v15 = 0;
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v52 + 536))(v4, 0, 4);
    v64 = -2;
  }
  else
  {
    v64 = -1;
  }
  v22 = *(_QWORD *)(v12 + 512);
  result = v22 + 24LL * *(unsigned int *)(v12 + 520);
  v59 = result;
  if ( result != v22 )
  {
    while ( 1 )
    {
      v37 = *(_QWORD *)(v22 + 16) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v22 + 4) )
        v24 = sub_3258B50(v37);
      else
        v24 = sub_2E309C0(v37, (__int64)v15, v19, v20, v21);
      v25 = *(_DWORD *)v22;
      HIBYTE(v67) = 1;
      v66[0] = "ToState";
      if ( v25 == -1 )
        v25 = v64;
      LOBYTE(v67) = 3;
      v26 = v25;
      v27 = *(_QWORD *)v4;
      if ( v10 )
        break;
      (*(void (__fastcall **)(__int64, __int64, __int64))(v27 + 536))(v4, v26, 4);
      v30 = *(_QWORD *)(v22 + 8);
      v31 = *(_QWORD *)(a1 + 8);
      if ( !v30 )
      {
LABEL_30:
        v33 = (unsigned __int8 *)sub_E81A90(0, *(_QWORD **)(v31 + 216), 0, 0);
        goto LABEL_21;
      }
LABEL_20:
      v32 = sub_31DB510(v31, v30);
      v33 = (unsigned __int8 *)sub_3258F50(a1, v32);
LABEL_21:
      sub_E9A5B0(v4, v33);
      v34 = *(_BYTE *)(v22 + 4) == 0;
      v35 = "ExceptionHandler";
      v67 = 259;
      if ( !v34 )
        v35 = "FinallyFunclet";
      v66[0] = v35;
      if ( v10 )
      {
        v36 = *(void (**)())(*(_QWORD *)v4 + 120LL);
        if ( v36 != nullsub_98 )
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v36)(v4, v66, 1);
      }
      v22 += 24;
      v15 = (unsigned __int8 *)sub_3258F50(a1, v24);
      result = sub_E9A5B0(v4, v15);
      if ( v59 == v22 )
        return result;
    }
    v28 = *(void (**)())(v27 + 120);
    if ( v28 != nullsub_98 )
    {
      v55 = v26;
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v28)(v4, v66, 1);
      v27 = *(_QWORD *)v4;
      v26 = v55;
    }
    (*(void (__fastcall **)(__int64, __int64, __int64))(v27 + 536))(v4, v26, 4);
    if ( *(_BYTE *)(v22 + 4) )
    {
      v66[0] = "Null";
      v67 = 259;
      v29 = *(void (**)())(*(_QWORD *)v4 + 120LL);
      if ( v29 == nullsub_98 )
        goto LABEL_19;
    }
    else
    {
      v66[0] = "FilterFunction";
      v67 = 259;
      v29 = *(void (**)())(*(_QWORD *)v4 + 120LL);
      if ( v29 == nullsub_98 )
        goto LABEL_19;
    }
    ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)()))v29)(v4, v66, 1, nullsub_98);
LABEL_19:
    v30 = *(_QWORD *)(v22 + 8);
    v31 = *(_QWORD *)(a1 + 8);
    if ( !v30 )
      goto LABEL_30;
    goto LABEL_20;
  }
  return result;
}
