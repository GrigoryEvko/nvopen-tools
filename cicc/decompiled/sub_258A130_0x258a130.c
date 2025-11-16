// Function: sub_258A130
// Address: 0x258a130
//
__int64 __fastcall sub_258A130(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r12
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rsi
  _BOOL4 v11; // eax
  __int64 v12; // r15
  __int64 v13; // r13
  bool (__fastcall *v14)(__int64); // rax
  __int64 (__fastcall *v15)(__int64); // rax
  int v17; // eax
  __int64 v18; // [rsp-10h] [rbp-60h]
  unsigned __int64 *v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v3 = a2;
  v19 = &a2[a3];
  if ( v19 == a2 )
    return 1;
  while ( 1 )
  {
    v5 = *v3;
    if ( **(_BYTE **)a1 == 3 )
    {
      sub_250D230(&v20, v5, 2, *(_QWORD *)(a1 + 8));
    }
    else
    {
      v21 = *(_QWORD *)(a1 + 8);
      v20 = v5 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
    }
    v6 = v20;
    v7 = sub_2589400(*(_QWORD *)(a1 + 16), v20, v21, *(_QWORD *)(a1 + 24), 0, 0, 1);
    v8 = v7;
    if ( !v7 )
      return 0;
    v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL);
    v10 = v9 == sub_2534AC0 ? v8 + 88 : ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v9)(v8, v6, v18);
    v11 = sub_255B670(*(_QWORD *)(a1 + 40), v10);
    sub_250C0C0(*(int **)(a1 + 32), v11);
    v12 = *(_QWORD *)(a1 + 40);
    v13 = *(_QWORD *)v12;
    v14 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v12 + 24LL);
    if ( v14 != sub_253A760 )
      break;
    if ( *(_DWORD *)(v12 + 24) <= 0x40u )
    {
      if ( *(_QWORD *)(v12 + 16) != *(_QWORD *)(v12 + 48) )
        goto LABEL_16;
      if ( *(_DWORD *)(v12 + 40) > 0x40u )
        goto LABEL_11;
LABEL_21:
      if ( *(_QWORD *)(v12 + 32) == *(_QWORD *)(v12 + 64) )
        goto LABEL_12;
      if ( v19 == ++v3 )
        return 1;
    }
    else
    {
      if ( !sub_C43C50(v12 + 16, (const void **)(v12 + 48)) )
        goto LABEL_16;
      if ( *(_DWORD *)(v12 + 40) <= 0x40u )
        goto LABEL_21;
LABEL_11:
      if ( sub_C43C50(v12 + 32, (const void **)(v12 + 64)) )
        goto LABEL_12;
LABEL_16:
      if ( v19 == ++v3 )
        return 1;
    }
  }
  if ( !v14(*(_QWORD *)(a1 + 40)) )
    goto LABEL_16;
  v12 = *(_QWORD *)(a1 + 40);
  v13 = *(_QWORD *)v12;
LABEL_12:
  v15 = *(__int64 (__fastcall **)(__int64))(v13 + 16);
  if ( v15 != sub_2535A50 )
    return v15(v12);
  if ( !*(_DWORD *)(v12 + 8) )
    return 0;
  LOBYTE(v17) = sub_AAF760(v12 + 16);
  return v17 ^ 1u;
}
