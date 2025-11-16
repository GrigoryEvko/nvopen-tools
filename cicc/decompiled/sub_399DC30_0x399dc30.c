// Function: sub_399DC30
// Address: 0x399dc30
//
void __fastcall sub_399DC30(__int64 a1)
{
  char *v1; // r9
  char *v2; // rdx
  __int64 v4; // rdi
  __int64 v5; // r10
  char *v6; // rax
  char *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int16 v12; // ax
  __int64 v13; // rdi
  __int64 v14; // r12
  void (__fastcall *v15)(__int64, _QWORD, _QWORD); // rbx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp-68h] [rbp-68h]
  __int64 v34; // [rsp-60h] [rbp-60h]
  _QWORD v35[2]; // [rsp-58h] [rbp-58h] BYREF
  char v36; // [rsp-48h] [rbp-48h]
  char v37; // [rsp-47h] [rbp-47h]

  v1 = *(char **)(a1 + 560);
  v2 = *(char **)(a1 + 552);
  if ( v2 == v1 )
    return;
  v4 = (v1 - v2) >> 6;
  v5 = (v1 - v2) >> 4;
  if ( v4 > 0 )
  {
    v6 = v2;
    while ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 1) + 80LL) + 36LL) == 3 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 3) + 80LL) + 36LL) != 3 )
      {
        if ( v1 != v6 + 16 )
          goto LABEL_10;
        return;
      }
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 5) + 80LL) + 36LL) != 3 )
      {
        if ( v1 != v6 + 32 )
          goto LABEL_10;
        return;
      }
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 7) + 80LL) + 36LL) != 3 )
      {
        if ( v1 != v6 + 48 )
          goto LABEL_10;
        return;
      }
      v6 += 64;
      if ( &v2[64 * v4] == v6 )
      {
        v28 = (v1 - v6) >> 4;
        goto LABEL_41;
      }
    }
    goto LABEL_9;
  }
  v28 = (v1 - v2) >> 4;
  v6 = v2;
LABEL_41:
  if ( v28 == 2 )
    goto LABEL_55;
  if ( v28 == 3 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 1) + 80LL) + 36LL) != 3 )
      goto LABEL_9;
    v6 += 16;
LABEL_55:
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 1) + 80LL) + 36LL) != 3 )
      goto LABEL_9;
    v6 += 16;
    goto LABEL_57;
  }
  if ( v28 != 1 )
    return;
LABEL_57:
  if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 1) + 80LL) + 36LL) == 3 )
    return;
LABEL_9:
  if ( v1 == v6 )
    return;
LABEL_10:
  if ( !*(_BYTE *)(a1 + 4501) )
    return;
  if ( v4 > 0 )
  {
    v7 = &v2[64 * v4];
    while ( 1 )
    {
      v11 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 616LL);
      if ( !v11 )
        v11 = *((_QWORD *)v2 + 1);
      if ( *(_DWORD *)(v11 + 744) )
        break;
      v8 = *(_QWORD *)(*((_QWORD *)v2 + 3) + 616LL);
      if ( !v8 )
        v8 = *((_QWORD *)v2 + 3);
      if ( *(_DWORD *)(v8 + 744) )
      {
        v2 += 16;
        break;
      }
      v9 = *(_QWORD *)(*((_QWORD *)v2 + 5) + 616LL);
      if ( !v9 )
        v9 = *((_QWORD *)v2 + 5);
      if ( *(_DWORD *)(v9 + 744) )
      {
        v2 += 32;
        break;
      }
      v10 = *(_QWORD *)(*((_QWORD *)v2 + 7) + 616LL);
      if ( !v10 )
        v10 = *((_QWORD *)v2 + 7);
      if ( *(_DWORD *)(v10 + 744) )
      {
        v2 += 48;
        break;
      }
      v2 += 64;
      if ( v7 == v2 )
      {
        v5 = (v1 - v2) >> 4;
        goto LABEL_62;
      }
    }
LABEL_26:
    if ( v1 == v2 )
      return;
    v12 = sub_398C0A0(a1);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(v13 + 256);
    v15 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v14 + 160LL);
    if ( v12 <= 4u )
    {
      v29 = sub_396DD80(v13);
      v15(v14, *(_QWORD *)(v29 + 160), 0);
      v22 = *(_QWORD *)(a1 + 552);
      v34 = *(_QWORD *)(a1 + 560);
      if ( v34 == v22 )
        return;
      v33 = 0;
    }
    else
    {
      v16 = sub_396DD80(v13);
      v17 = a1 + 4040;
      v15(v14, *(_QWORD *)(v16 + 296), 0);
      if ( *(_BYTE *)(a1 + 4513) )
        v17 = a1 + 4520;
      v18 = *(_QWORD *)(a1 + 8);
      v37 = 1;
      v35[0] = "debug_rnglist_table_start";
      v36 = 3;
      v19 = sub_396F530(v18, (__int64)v35);
      v37 = 1;
      v20 = v19;
      v36 = 3;
      v35[0] = "debug_rnglist_table_end";
      v33 = sub_396F530(v18, (__int64)v35);
      sub_396F380(v18);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v18 + 256) + 176LL))(
        *(_QWORD *)(v18 + 256),
        v20,
        0);
      sub_396F320(v18, *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v18 + 256) + 8LL) + 1160LL));
      sub_396F300(v18, *(_DWORD *)(*(_QWORD *)(v18 + 240) + 8LL));
      sub_396F300(v18, 0);
      v21 = *(_QWORD *)(v17 + 256);
      sub_396F340(v18, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v18 + 256) + 176LL))(
        *(_QWORD *)(v18 + 256),
        v21,
        0);
      v22 = *(_QWORD *)(a1 + 552);
      v34 = *(_QWORD *)(a1 + 560);
      if ( v22 == v34 )
        goto LABEL_38;
    }
    do
    {
      v23 = *(_QWORD *)(v22 + 8);
      if ( *(_DWORD *)(*(_QWORD *)(v23 + 80) + 36LL) != 3 )
      {
        v24 = *(_QWORD *)(v23 + 616);
        if ( v24 )
        {
          v23 = *(_QWORD *)(v24 + 616);
          if ( !v23 )
            v23 = v24;
        }
        else
        {
          v24 = *(_QWORD *)(v22 + 8);
        }
        v25 = *(_QWORD *)(v23 + 736);
        v26 = v25 + 56LL * *(unsigned int *)(v23 + 744);
        while ( v26 != v25 )
        {
          v27 = v25;
          v25 += 56;
          sub_399D1D0(*(_QWORD *)(a1 + 8), v24, v27);
        }
      }
      v22 += 16;
    }
    while ( v22 != v34 );
LABEL_38:
    if ( v33 )
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
        v33,
        0);
    return;
  }
LABEL_62:
  if ( v5 == 2 )
    goto LABEL_76;
  if ( v5 == 3 )
  {
    v31 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 616LL);
    if ( !v31 )
      v31 = *((_QWORD *)v2 + 1);
    if ( *(_DWORD *)(v31 + 744) )
      goto LABEL_26;
    v2 += 16;
LABEL_76:
    v32 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 616LL);
    if ( !v32 )
      v32 = *((_QWORD *)v2 + 1);
    if ( *(_DWORD *)(v32 + 744) )
      goto LABEL_26;
    v2 += 16;
    goto LABEL_65;
  }
  if ( v5 != 1 )
    return;
LABEL_65:
  v30 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 616LL);
  if ( !v30 )
    v30 = *((_QWORD *)v2 + 1);
  if ( *(_DWORD *)(v30 + 744) )
    goto LABEL_26;
}
