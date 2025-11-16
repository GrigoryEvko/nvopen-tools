// Function: sub_39AF220
// Address: 0x39af220
//
__int64 __fastcall sub_39AF220(__int64 a1, __int64 *a2)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // r8
  char v10; // r15
  __int64 (*v11)(); // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdi
  const char *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 result; // rax
  __int64 v21; // r13
  int v22; // eax
  __int64 v23; // r8
  __int64 v24; // rax
  void (*v25)(); // r9
  void (*v26)(); // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned int *v30; // rsi
  bool v31; // zf
  const char *v32; // rax
  void (*v33)(); // rax
  unsigned int *v34; // rax
  unsigned __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  __int64 v40; // rax
  int v41; // eax
  int v42; // ebx
  __int64 (*v43)(); // rax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // r13
  __int64 v47; // r8
  __int64 v48; // rax
  void (*v49)(); // rcx
  __int64 v50; // r9
  __int64 v51; // rax
  void (*v52)(); // rcx
  __int64 v53; // r9
  __int64 v54; // rax
  void (*v55)(); // rcx
  __int64 v56; // r9
  __int64 v57; // rax
  void (*v58)(); // rcx
  char v59; // al
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+8h] [rbp-68h]
  __int64 v62; // [rsp+10h] [rbp-60h]
  __int64 v63; // [rsp+10h] [rbp-60h]
  __int64 v64; // [rsp+10h] [rbp-60h]
  __int64 v65; // [rsp+10h] [rbp-60h]
  __int64 v66; // [rsp+18h] [rbp-58h]
  __int64 v67; // [rsp+18h] [rbp-58h]
  int v68; // [rsp+18h] [rbp-58h]
  __int64 v69; // [rsp+18h] [rbp-58h]
  __int64 v70; // [rsp+18h] [rbp-58h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  __int64 v72; // [rsp+18h] [rbp-58h]
  __int64 v73; // [rsp+18h] [rbp-58h]
  __int64 v74; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  _QWORD v76[2]; // [rsp+20h] [rbp-50h] BYREF
  char v77; // [rsp+30h] [rbp-40h]
  char v78; // [rsp+31h] [rbp-3Fh]

  v4 = *a2;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v6 = sub_1649960(*a2);
  v8 = (__int64)v6;
  v9 = v7;
  if ( v7 && *v6 == 1 )
  {
    v9 = v7 - 1;
    v8 = (__int64)(v6 + 1);
  }
  v10 = 0;
  v11 = *(__int64 (**)())(*(_QWORD *)v5 + 80LL);
  if ( v11 != sub_168DB50 )
  {
    v65 = v8;
    v73 = v9;
    v59 = ((__int64 (__fastcall *)(__int64))v11)(v5);
    v8 = v65;
    v9 = v73;
    v10 = v59;
  }
  v60 = v8;
  v62 = a2[11];
  v66 = v9;
  sub_39AEF10(a1, v62, v8, v9);
  v67 = sub_38BF870(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL), v60, v66);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v5 + 512LL))(v5, 4, 0, 1, 0);
  v12 = v67;
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v5 + 176LL))(v5, v67, 0);
  v13 = sub_15E38F0(v4);
  v14 = sub_1649C60(v13);
  if ( *(_BYTE *)(v14 + 16) )
    v14 = 0;
  v15 = sub_1649960(v14);
  v68 = -1;
  v17 = v62;
  if ( v18 == 16 )
  {
    v36 = *(_QWORD *)v15 ^ 0x5F7470656378655FLL;
    if ( !(v36 | *((_QWORD *)v15 + 1) ^ 0x3472656C646E6168LL) )
    {
      v37 = a2[7];
      if ( *(_DWORD *)(v37 + 68) == -1 )
      {
        v42 = -2;
      }
      else
      {
        v38 = a2[2];
        v39 = *(__int64 (**)())(*(_QWORD *)v38 + 48LL);
        if ( v39 == sub_1D90020 )
          BUG();
        v40 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64, __int64))v39)(
                v38,
                v12,
                v36,
                0,
                v16,
                v62);
        v41 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD, _QWORD *))(*(_QWORD *)v40 + 176LL))(
                v40,
                a2,
                *(unsigned int *)(v37 + 68),
                v76);
        v17 = v62;
        v42 = v41;
      }
      if ( *(_DWORD *)(v17 + 720) == 0x7FFFFFFF )
      {
        v46 = 9999;
      }
      else
      {
        v69 = v17;
        v43 = *(__int64 (**)())(*(_QWORD *)a2[2] + 48LL);
        if ( v43 == sub_1D90020 )
          BUG();
        v44 = v43();
        v45 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD, _QWORD *))(*(_QWORD *)v44 + 176LL))(
                v44,
                a2,
                *(unsigned int *)(v69 + 720),
                v76);
        v17 = v69;
        v46 = v45;
      }
      v78 = 1;
      v47 = v42;
      v76[0] = "GSCookieOffset";
      v77 = 3;
      if ( v10 )
      {
        v48 = *(_QWORD *)v5;
        v49 = *(void (**)())(*(_QWORD *)v5 + 104LL);
        if ( v49 != nullsub_580 )
        {
          v75 = v17;
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v49)(v5, v76, 1);
          v48 = *(_QWORD *)v5;
          v17 = v75;
          v47 = v42;
        }
        v70 = v17;
        (*(void (__fastcall **)(__int64, __int64, __int64))(v48 + 424))(v5, v47, 4);
        v78 = 1;
        v50 = v70;
        v76[0] = "GSCookieXOROffset";
        v77 = 3;
        v51 = *(_QWORD *)v5;
        v52 = *(void (**)())(*(_QWORD *)v5 + 104LL);
        if ( v52 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v52)(v5, v76, 1);
          v51 = *(_QWORD *)v5;
          v50 = v70;
        }
        v71 = v50;
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v51 + 424))(v5, 0, 4);
        v78 = 1;
        v53 = v71;
        v76[0] = "EHCookieOffset";
        v77 = 3;
        v54 = *(_QWORD *)v5;
        v55 = *(void (**)())(*(_QWORD *)v5 + 104LL);
        if ( v55 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v55)(v5, v76, 1);
          v54 = *(_QWORD *)v5;
          v53 = v71;
        }
        v72 = v53;
        (*(void (__fastcall **)(__int64, __int64, __int64))(v54 + 424))(v5, v46, 4);
        v78 = 1;
        v56 = v72;
        v76[0] = "EHCookieXOROffset";
        v77 = 3;
        v57 = *(_QWORD *)v5;
        v58 = *(void (**)())(*(_QWORD *)v5 + 104LL);
        if ( v58 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v58)(v5, v76, 1);
          v57 = *(_QWORD *)v5;
          v56 = v72;
        }
      }
      else
      {
        v74 = v17;
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v5 + 424LL))(v5, v42, 4);
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v5 + 424LL))(v5, 0, 4);
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 424LL))(v5, v46, 4);
        v57 = *(_QWORD *)v5;
        v56 = v74;
      }
      v64 = v56;
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(v57 + 424))(v5, 0, 4);
      v68 = -2;
      v17 = v64;
    }
  }
  v19 = *(_QWORD *)(v17 + 480);
  result = v19 + 24LL * *(unsigned int *)(v17 + 488);
  v63 = result;
  if ( result != v19 )
  {
    while ( 1 )
    {
      v35 = *(_QWORD *)(v19 + 16) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v19 + 4) )
        v21 = sub_39AC850(v35);
      else
        v21 = sub_1DD5A70(v35);
      v22 = *(_DWORD *)v19;
      v78 = 1;
      v76[0] = "ToState";
      if ( v22 == -1 )
        v22 = v68;
      v77 = 3;
      v23 = v22;
      v24 = *(_QWORD *)v5;
      if ( v10 )
        break;
      (*(void (__fastcall **)(__int64, __int64, __int64))(v24 + 424))(v5, v23, 4);
      v27 = *(_QWORD *)(v19 + 8);
      v28 = *(_QWORD *)(a1 + 8);
      if ( !v27 )
      {
LABEL_30:
        v30 = (unsigned int *)sub_38CB470(0, *(_QWORD *)(v28 + 248));
        goto LABEL_21;
      }
LABEL_20:
      v29 = sub_396EAF0(v28, v27);
      v30 = (unsigned int *)sub_39ACBF0(a1, v29);
LABEL_21:
      sub_38DDD30(v5, v30);
      v31 = *(_BYTE *)(v19 + 4) == 0;
      v78 = 1;
      v32 = "ExceptionHandler";
      v77 = 3;
      if ( !v31 )
        v32 = "FinallyFunclet";
      v76[0] = v32;
      if ( v10 )
      {
        v33 = *(void (**)())(*(_QWORD *)v5 + 104LL);
        if ( v33 != nullsub_580 )
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v33)(v5, v76, 1);
      }
      v19 += 24;
      v34 = (unsigned int *)sub_39ACBF0(a1, v21);
      result = sub_38DDD30(v5, v34);
      if ( v63 == v19 )
        return result;
    }
    v25 = *(void (**)())(v24 + 104);
    if ( v25 != nullsub_580 )
    {
      v61 = v23;
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v5, v76, 1);
      v24 = *(_QWORD *)v5;
      v23 = v61;
    }
    (*(void (__fastcall **)(__int64, __int64, __int64))(v24 + 424))(v5, v23, 4);
    if ( *(_BYTE *)(v19 + 4) )
    {
      v78 = 1;
      v76[0] = "Null";
      v77 = 3;
      v26 = *(void (**)())(*(_QWORD *)v5 + 104LL);
      if ( v26 == nullsub_580 )
        goto LABEL_19;
    }
    else
    {
      v78 = 1;
      v76[0] = "FilterFunction";
      v77 = 3;
      v26 = *(void (**)())(*(_QWORD *)v5 + 104LL);
      if ( v26 == nullsub_580 )
        goto LABEL_19;
    }
    ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)()))v26)(v5, v76, 1, nullsub_580);
LABEL_19:
    v27 = *(_QWORD *)(v19 + 8);
    v28 = *(_QWORD *)(a1 + 8);
    if ( !v27 )
      goto LABEL_30;
    goto LABEL_20;
  }
  return result;
}
