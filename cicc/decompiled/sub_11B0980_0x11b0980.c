// Function: sub_11B0980
// Address: 0x11b0980
//
_QWORD *__fastcall sub_11B0980(__int64 a1, __int64 a2)
{
  _DWORD *v4; // rax
  _DWORD *v5; // rcx
  unsigned __int8 *v6; // r14
  __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 v10; // rcx
  __int64 v11; // rax
  _DWORD *v12; // rdi
  _DWORD *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r10
  __int64 v16; // r11
  __int64 *v17; // r13
  unsigned int v18; // r12d
  unsigned __int8 *v19; // r15
  __int64 v20; // r12
  void *v21; // r14
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  int v25; // r12d
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  _DWORD *v30; // rdi
  _DWORD *v31; // rsi
  bool v32; // zf
  __int64 v33; // [rsp-10h] [rbp-C0h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  __int64 v35; // [rsp+18h] [rbp-98h]
  char v36[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v37; // [rsp+40h] [rbp-70h]
  _BYTE v38[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v39; // [rsp+70h] [rbp-40h]

  if ( **(_BYTE **)(a2 - 32) != 13 )
    return 0;
  v4 = sub_11AECA0(*(_DWORD **)(a2 + 72), *(_QWORD *)(a2 + 72) + 4LL * *(unsigned int *)(a2 + 80));
  if ( v5 != v4 )
    return 0;
  v6 = *(unsigned __int8 **)(a2 - 64);
  v7 = *((_QWORD *)v6 + 2);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD **)(v7 + 8);
  if ( v8 )
    return 0;
  if ( (unsigned __int8)(*v6 - 42) > 0x11u )
    return v8;
  v10 = *((_QWORD *)v6 - 8);
  if ( *(_BYTE *)v10 == 92 && *(_QWORD *)(v10 - 64) && **(_BYTE **)(v10 - 32) == 13 )
  {
    v30 = *(_DWORD **)(v10 + 72);
    v31 = &v30[*(unsigned int *)(v10 + 80)];
    v32 = v31 == sub_11AECA0(v30, (__int64)v31);
    v11 = *((_QWORD *)v6 - 4);
    if ( v32 )
    {
      v15 = *((_QWORD *)v6 - 4);
      if ( v11 )
        goto LABEL_17;
    }
  }
  else
  {
    v11 = *((_QWORD *)v6 - 4);
  }
  if ( *(_BYTE *)v11 != 92 )
    return v8;
  if ( !*(_QWORD *)(v11 - 64) )
    return v8;
  if ( **(_BYTE **)(v11 - 32) != 13 )
    return v8;
  v12 = *(_DWORD **)(v11 + 72);
  v13 = &v12[*(unsigned int *)(v11 + 80)];
  if ( v13 != sub_11AECA0(v12, (__int64)v13) )
    return v8;
  v16 = v14;
LABEL_17:
  v34 = v16;
  v35 = v15;
  if ( *(_QWORD *)(v15 + 8) != *(_QWORD *)(v16 + 8) || !sub_991A70(v6, 0, 0, 0, 0, 0, 0) )
    return 0;
  v17 = *(__int64 **)(a1 + 32);
  v37 = 257;
  v18 = *v6 - 29;
  v19 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v17[10]
                                                                                                 + 16LL))(
                             v17[10],
                             v18,
                             v34,
                             v35,
                             v33);
  if ( !v19 )
  {
    v39 = 257;
    v19 = (unsigned __int8 *)sub_B504D0(v18, v34, v35, (__int64)v38, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v19) )
    {
      v24 = v17[12];
      v25 = *((_DWORD *)v17 + 26);
      if ( v24 )
        sub_B99FD0((__int64)v19, 3u, v24);
      sub_B45150((__int64)v19, v25);
    }
    (*(void (__fastcall **)(__int64, unsigned __int8 *, char *, __int64, __int64))(*(_QWORD *)v17[11] + 16LL))(
      v17[11],
      v19,
      v36,
      v17[7],
      v17[8]);
    v26 = *v17;
    v27 = *v17 + 16LL * *((unsigned int *)v17 + 2);
    while ( v27 != v26 )
    {
      v28 = *(_QWORD *)(v26 + 8);
      v29 = *(_DWORD *)v26;
      v26 += 16;
      sub_B99FD0((__int64)v19, v29, v28);
    }
  }
  if ( *v19 > 0x1Cu )
    sub_B45260(v19, (__int64)v6, 1);
  v20 = *(unsigned int *)(a2 + 80);
  v39 = 257;
  v21 = *(void **)(a2 + 72);
  v22 = sub_BD2C40(112, unk_3F1FE60);
  v8 = v22;
  if ( v22 )
    sub_B4EB40((__int64)v22, (__int64)v19, v21, v20, (__int64)v38, v23, 0);
  return v8;
}
