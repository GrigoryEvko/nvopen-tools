// Function: sub_2E7F230
// Address: 0x2e7f230
//
__int64 __fastcall sub_2E7F230(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 (*v7)(void); // rax
  __int64 v8; // rax
  __int64 (__fastcall *v9)(__int64); // rcx
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // edx
  int *v16; // rcx
  int v17; // esi
  __int64 result; // rax
  int v19; // ecx
  unsigned int v20; // esi
  __int64 v21; // r9
  unsigned int v22; // r8d
  _DWORD *v23; // rcx
  int v24; // edx
  int v25; // r11d
  _DWORD *v26; // rdi
  int v27; // ecx
  int v28; // ecx
  __int64 v29; // rdx
  int v30; // r9d
  int v31; // edx
  int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // edx
  int v35; // r8d
  int v36; // r11d
  _DWORD *v37; // r10
  int v38; // edx
  int v39; // edx
  __int64 v40; // r9
  __int64 v41; // r12
  int v42; // r8d
  int v43; // esi
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  _QWORD v46[2]; // [rsp+10h] [rbp-60h] BYREF
  char v47; // [rsp+20h] [rbp-50h]
  __int64 v48; // [rsp+30h] [rbp-40h] BYREF
  char v49; // [rsp+40h] [rbp-30h]

  v4 = 0;
  v7 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 128LL);
  if ( v7 != sub_2DAC790 )
    v4 = v7();
  if ( *(_WORD *)(a2 + 68) == 20 )
    goto LABEL_6;
  v8 = *(_QWORD *)v4;
  v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 520LL);
  if ( v9 == sub_2DCA430 )
  {
LABEL_5:
    v10 = *(__int64 (__fastcall **)(__int64))(v8 + 528);
    if ( v10 != sub_2E77FE0 )
    {
      ((void (__fastcall *)(_QWORD *, __int64, unsigned __int64))v10)(v46, v4, a2);
      if ( v47 )
      {
        v11 = v46[0];
        goto LABEL_7;
      }
    }
LABEL_6:
    v11 = *(_QWORD *)(a2 + 32);
    goto LABEL_7;
  }
  ((void (__fastcall *)(__int64 *, __int64, unsigned __int64))v9)(&v48, v4, a2);
  v11 = v48;
  if ( !v49 )
  {
    v8 = *(_QWORD *)v4;
    goto LABEL_5;
  }
LABEL_7:
  v12 = *(_DWORD *)(v11 + 8);
  v13 = *(unsigned int *)(a3 + 24);
  v14 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v13 )
  {
    v15 = (v13 - 1) & (37 * v12);
    v16 = (int *)(v14 + 12LL * v15);
    v17 = *v16;
    if ( v12 == *v16 )
    {
LABEL_9:
      if ( v16 != (int *)(v14 + 12 * v13) )
        return *(_QWORD *)(v16 + 1);
    }
    else
    {
      v19 = 1;
      while ( v17 != -1 )
      {
        v30 = v19 + 1;
        v15 = (v13 - 1) & (v19 + v15);
        v16 = (int *)(v14 + 12LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
          goto LABEL_9;
        v19 = v30;
      }
    }
  }
  result = sub_2E7B480(a1, a2);
  v20 = *(_DWORD *)(a3 + 24);
  *(_QWORD *)((char *)v46 + 4) = result;
  if ( !v20 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_32;
  }
  v21 = *(_QWORD *)(a3 + 8);
  v22 = (v20 - 1) & (37 * v12);
  v23 = (_DWORD *)(v21 + 12LL * v22);
  v24 = *v23;
  if ( v12 == *v23 )
    return result;
  v25 = 1;
  v26 = 0;
  while ( v24 != -1 )
  {
    if ( v24 == -2 && !v26 )
      v26 = v23;
    v22 = (v20 - 1) & (v25 + v22);
    v23 = (_DWORD *)(v21 + 12LL * v22);
    v24 = *v23;
    if ( v12 == *v23 )
      return result;
    ++v25;
  }
  if ( !v26 )
    v26 = v23;
  v27 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v28 = v27 + 1;
  if ( 4 * v28 >= 3 * v20 )
  {
LABEL_32:
    v44 = result;
    sub_2E7F050(a3, 2 * v20);
    v31 = *(_DWORD *)(a3 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a3 + 8);
      v34 = (v31 - 1) & (37 * v12);
      v26 = (_DWORD *)(v33 + 12LL * v34);
      v28 = *(_DWORD *)(a3 + 16) + 1;
      result = v44;
      v35 = *v26;
      if ( v12 == *v26 )
        goto LABEL_24;
      v36 = 1;
      v37 = 0;
      while ( v35 != -1 )
      {
        if ( v35 == -2 && !v37 )
          v37 = v26;
        v34 = v32 & (v36 + v34);
        v26 = (_DWORD *)(v33 + 12LL * v34);
        v35 = *v26;
        if ( v12 == *v26 )
          goto LABEL_24;
        ++v36;
      }
LABEL_36:
      if ( v37 )
        v26 = v37;
      goto LABEL_24;
    }
LABEL_57:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
  if ( v20 - *(_DWORD *)(a3 + 20) - v28 <= v20 >> 3 )
  {
    v45 = result;
    sub_2E7F050(a3, v20);
    v38 = *(_DWORD *)(a3 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a3 + 8);
      v37 = 0;
      LODWORD(v41) = v39 & (37 * v12);
      v42 = 1;
      v26 = (_DWORD *)(v40 + 12LL * (unsigned int)v41);
      v28 = *(_DWORD *)(a3 + 16) + 1;
      result = v45;
      v43 = *v26;
      if ( v12 == *v26 )
        goto LABEL_24;
      while ( v43 != -1 )
      {
        if ( !v37 && v43 == -2 )
          v37 = v26;
        v41 = v39 & (unsigned int)(v41 + v42);
        v26 = (_DWORD *)(v40 + 12 * v41);
        v43 = *v26;
        if ( v12 == *v26 )
          goto LABEL_24;
        ++v42;
      }
      goto LABEL_36;
    }
    goto LABEL_57;
  }
LABEL_24:
  *(_DWORD *)(a3 + 16) = v28;
  if ( *v26 != -1 )
    --*(_DWORD *)(a3 + 20);
  v29 = *(_QWORD *)((char *)v46 + 4);
  *v26 = v12;
  *(_QWORD *)(v26 + 1) = v29;
  return result;
}
