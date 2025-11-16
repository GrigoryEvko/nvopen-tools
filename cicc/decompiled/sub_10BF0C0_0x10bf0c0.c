// Function: sub_10BF0C0
// Address: 0x10bf0c0
//
unsigned __int8 *__fastcall sub_10BF0C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int8 *v6; // r12
  __int64 v8; // rbx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 *v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r15
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r14
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rax
  const char *v30; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+8h] [rbp-B8h]
  char *v32; // [rsp+10h] [rbp-B0h]
  __int16 v33; // [rsp+20h] [rbp-A0h]
  const char *v34; // [rsp+30h] [rbp-90h] BYREF
  __int16 v35; // [rsp+50h] [rbp-70h]
  _BYTE v36[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v37; // [rsp+80h] [rbp-40h]

  v3 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 16LL);
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_QWORD *)(v3 + 8) )
  {
    v5 = *(_QWORD *)(v4 + 16);
    if ( !v5 || *(_QWORD *)(v5 + 8) )
      return 0;
  }
  if ( *(_BYTE *)v4 != 69 )
    return 0;
  v8 = *(_QWORD *)(v4 - 32);
  if ( !v8 )
    return 0;
  v9 = *(_QWORD *)(v8 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  if ( !sub_BCAC40(v9, 1) )
    return 0;
  v10 = *(_BYTE **)(a2 - 64);
  if ( *v10 != 42 )
    return 0;
  v11 = (_BYTE *)*((_QWORD *)v10 - 8);
  v12 = *((_QWORD *)v10 - 4);
  if ( *v11 == 69 )
  {
    v29 = *((_QWORD *)v11 - 4);
    if ( v29 )
    {
      if ( v8 == v29 && v12 )
        goto LABEL_17;
    }
  }
  if ( *(_BYTE *)v12 != 69 )
    return 0;
  v13 = *(_QWORD *)(v12 - 32);
  if ( !v13 || v8 != v13 )
    return 0;
  v12 = (__int64)v11;
LABEL_17:
  v14 = *(__int64 **)(a1 + 32);
  v35 = 257;
  v30 = sub_BD5D20(v12);
  v33 = 773;
  v31 = v15;
  v32 = ".neg";
  v16 = sub_AD6530(*(_QWORD *)(v12 + 8), 1);
  v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD, const char *, __int64, char *))(*(_QWORD *)v14[10] + 32LL))(
          v14[10],
          15,
          v16,
          v12,
          0,
          0,
          v30,
          v31,
          ".neg");
  if ( !v17 )
  {
    v37 = 257;
    v17 = sub_B504D0(15, v16, v12, (__int64)v36, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v14[11] + 16LL))(
      v14[11],
      v17,
      &v30,
      v14[7],
      v14[8]);
    v25 = *v14;
    v26 = *v14 + 16LL * *((unsigned int *)v14 + 2);
    while ( v26 != v25 )
    {
      v27 = *(_QWORD *)(v25 + 8);
      v28 = *(_DWORD *)v25;
      v25 += 16;
      sub_B99FD0(v17, v28, v27);
    }
  }
  v18 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v6 = v18;
  if ( v18 )
  {
    sub_B44260((__int64)v18, *(_QWORD *)(v17 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v6 - 12) )
    {
      v19 = *((_QWORD *)v6 - 11);
      **((_QWORD **)v6 - 10) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *((_QWORD *)v6 - 10);
    }
    *((_QWORD *)v6 - 12) = v8;
    v20 = *(_QWORD *)(v8 + 16);
    *((_QWORD *)v6 - 11) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v6 - 88;
    *((_QWORD *)v6 - 10) = v8 + 16;
    *(_QWORD *)(v8 + 16) = v6 - 96;
    if ( *((_QWORD *)v6 - 8) )
    {
      v21 = *((_QWORD *)v6 - 7);
      **((_QWORD **)v6 - 6) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = *((_QWORD *)v6 - 6);
    }
    *((_QWORD *)v6 - 8) = v17;
    v22 = *(_QWORD *)(v17 + 16);
    *((_QWORD *)v6 - 7) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = v6 - 56;
    *((_QWORD *)v6 - 6) = v17 + 16;
    *(_QWORD *)(v17 + 16) = v6 - 64;
    if ( *((_QWORD *)v6 - 4) )
    {
      v23 = *((_QWORD *)v6 - 3);
      **((_QWORD **)v6 - 2) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *((_QWORD *)v6 - 2);
    }
    *((_QWORD *)v6 - 4) = v12;
    v24 = *(_QWORD *)(v12 + 16);
    *((_QWORD *)v6 - 3) = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = v6 - 24;
    *((_QWORD *)v6 - 2) = v12 + 16;
    *(_QWORD *)(v12 + 16) = v6 - 32;
    sub_BD6B50(v6, &v34);
  }
  return v6;
}
