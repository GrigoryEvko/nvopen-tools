// Function: sub_1EAD840
// Address: 0x1ead840
//
__int64 __fastcall sub_1EAD840(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v4; // r13
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 (*v7)(void); // rdx
  __int64 (*v8)(void); // rdx
  __int64 (*v9)(void); // rax
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 result; // rax
  _DWORD *v14; // r14
  int v15; // edx
  int v16; // eax
  int v17; // ebx
  __int64 v18; // r15
  int v19; // r13d
  __int64 v20; // rax
  __int64 v21; // rcx
  __int16 v22; // si
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rbx
  __int64 i; // rbx
  __int64 v26; // rdx
  void (__fastcall *v27)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v28; // rbx
  __int64 v29; // r13
  _QWORD *v30; // rax
  int v31; // eax
  __int64 v32; // rdx
  void (__fastcall *v33)(__int64, __int64, __int64, __int64, _QWORD); // rax
  __int64 v34; // rdx
  __int64 (__fastcall *v35)(__int64); // rax
  int v36; // eax
  __int64 v37; // r9
  __int64 v38; // r10
  int v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  bool v42; // [rsp+27h] [rbp-79h]
  __int64 v43; // [rsp+28h] [rbp-78h]
  __int64 v44; // [rsp+30h] [rbp-70h]
  __int64 v47; // [rsp+48h] [rbp-58h]
  int v50[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v4 = 0;
  v5 = *(__int64 **)(a3 + 16);
  v6 = *v5;
  v7 = *(__int64 (**)(void))(*v5 + 40);
  if ( v7 != sub_1D00B00 )
  {
    v4 = v7();
    v6 = **(_QWORD **)(a3 + 16);
  }
  v8 = *(__int64 (**)(void))(v6 + 112);
  v43 = 0;
  if ( v8 != sub_1D00B10 )
  {
    v43 = v8();
    v6 = **(_QWORD **)(a3 + 16);
  }
  v9 = *(__int64 (**)(void))(v6 + 48);
  v10 = 0;
  if ( v9 != sub_1D90020 )
    v10 = v9();
  v11 = *(_QWORD *)(a1 + 232);
  if ( v11 && *(_BYTE *)(a1 + 345) )
    sub_1EEA020(v11, a2);
  v42 = 0;
  v12 = *(_QWORD *)(a2 + 32);
  result = a2 + 24;
  v47 = a2 + 24;
  if ( v12 == a2 + 24 )
    return result;
  v44 = v10;
  v14 = (_DWORD *)v4;
  while ( 1 )
  {
    while ( 1 )
    {
      v15 = v14[9];
      v16 = **(unsigned __int16 **)(v12 + 16);
      if ( v15 != v16 && v14[10] != v16 )
        break;
      v42 = v15 == v16;
      *a4 += (*(__int64 (__fastcall **)(_DWORD *, __int64))(*(_QWORD *)v14 + 32LL))(v14, v12);
      result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v44 + 216LL))(
                 v44,
                 a3,
                 a2,
                 v12);
      v12 = result;
LABEL_40:
      if ( v12 == v47 )
        return result;
    }
    if ( *(_DWORD *)(v12 + 40) )
      break;
LABEL_59:
    if ( v42 )
      *a4 += (*(__int64 (__fastcall **)(_DWORD *, __int64))(*(_QWORD *)v14 + 32LL))(v14, v12);
    result = 1;
    if ( v12 == v47 )
      goto LABEL_46;
    v24 = v12;
    result = 1;
    if ( (*(_BYTE *)v12 & 4) != 0 )
      goto LABEL_30;
LABEL_66:
    while ( (*(_BYTE *)(v24 + 46) & 8) != 0 )
      v24 = *(_QWORD *)(v24 + 8);
LABEL_30:
    v28 = *(_QWORD *)(v24 + 8);
    v29 = *(_QWORD *)(a1 + 232);
    if ( v29 && *(_BYTE *)(a1 + 345) && (_BYTE)result )
    {
LABEL_50:
      if ( !*(_BYTE *)(v29 + 44) )
      {
        result = *(_QWORD *)(v29 + 24);
        if ( *(_QWORD *)(result + 32) != v12 )
          result = sub_1EEA3B0(v29);
      }
      if ( v12 != *(_QWORD *)(v29 + 32) )
      {
        do
          result = sub_1EEA3B0(v29);
        while ( *(_QWORD *)(v29 + 32) != v12 );
        v12 = v28;
        goto LABEL_34;
      }
    }
    v12 = v28;
LABEL_34:
    if ( v12 == v47 )
      return result;
  }
  v17 = 1;
  v18 = 0;
  v19 = *(_DWORD *)(v12 + 40);
  while ( 1 )
  {
    v20 = *(_QWORD *)(v12 + 32);
    v21 = (unsigned int)(v17 - 1);
    if ( *(_BYTE *)(v20 + v18) != 5 )
      goto LABEL_37;
    v22 = **(_WORD **)(v12 + 16);
    if ( v22 == 12 )
    {
      v39 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, int *))(*(_QWORD *)v44 + 176LL))(
              v44,
              a3,
              *(unsigned int *)(v20 + 24),
              v50);
      sub_1E31400(*(char **)(v12 + 32), v50[0], 0, 0, 0, 0, 0, 0);
      *(_BYTE *)(*(_QWORD *)(v12 + 32) + 4LL) |= 8u;
      v30 = (_QWORD *)sub_1E16510(v12);
      *(_QWORD *)(*(_QWORD *)(v12 + 32) + 144LL) = sub_15C48E0(v30, 0, v39, 0, 0);
LABEL_37:
      v31 = v17 + 1;
      v18 += 40;
      if ( v19 == v17 )
        goto LABEL_59;
      goto LABEL_38;
    }
    if ( v22 != 23 )
      break;
    v34 = *(unsigned int *)(v20 + v18 + 24);
    v40 = v20 + v18 + 40;
    v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v44 + 184LL);
    if ( v35 == sub_1EAD650 )
    {
      v36 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v44 + 176LL))(v44, a3, v34, v50);
      v37 = v40;
      v38 = v18 + 40;
    }
    else
    {
      v36 = ((__int64 (__fastcall *)(__int64, __int64, __int64, int *, _QWORD))v35)(v44, a3, v34, v50, 0);
      v38 = v18 + 40;
      v37 = v40;
    }
    *(_QWORD *)(v37 + 24) += v36;
    v41 = v38;
    sub_1E31400((char *)(v18 + *(_QWORD *)(v12 + 32)), v50[0], 0, 0, 0, 0, 0, 0);
    v31 = v17 + 1;
    v18 = v41;
    if ( v19 == v17 )
      goto LABEL_59;
LABEL_38:
    v17 = v31;
  }
  if ( *(_QWORD *)(a2 + 32) == v12 )
  {
    v32 = (unsigned int)*a4;
    v33 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v43 + 392LL);
    if ( *(_BYTE *)(a1 + 345) )
      v33(v43, v12, v32, v21, *(_QWORD *)(a1 + 232));
    else
      v33(v43, v12, v32, v21, 0);
    result = a2;
    v12 = *(_QWORD *)(a2 + 32);
    goto LABEL_40;
  }
  v23 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v23 )
    BUG();
  v24 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)v23 & 4) == 0 && (*(_BYTE *)(v23 + 46) & 4) != 0 )
  {
    for ( i = *(_QWORD *)v23; ; i = *(_QWORD *)v24 )
    {
      v24 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v24 + 46) & 4) == 0 )
        break;
    }
  }
  v26 = (unsigned int)*a4;
  v27 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v43 + 392LL);
  if ( *(_BYTE *)(a1 + 345) )
    v27(v43, v12, v26, v21, *(_QWORD *)(a1 + 232));
  else
    v27(v43, v12, v26, v21, 0);
  result = 0;
  if ( v24 != v47 )
  {
    if ( (*(_BYTE *)v24 & 4) != 0 )
      goto LABEL_30;
    goto LABEL_66;
  }
LABEL_46:
  v29 = *(_QWORD *)(a1 + 232);
  if ( v29 && *(_BYTE *)(a1 + 345) && (_BYTE)result )
  {
    v28 = v47;
    goto LABEL_50;
  }
  return result;
}
