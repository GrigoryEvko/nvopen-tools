// Function: sub_257D2E0
// Address: 0x257d2e0
//
char __fastcall sub_257D2E0(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v4; // r12
  unsigned __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rdi
  _DWORD *v10; // r15
  __int64 v11; // rsi
  int v12; // eax
  void (__fastcall *v13)(__int64, int); // rdx
  int v14; // esi
  _DWORD *v15; // rdi
  bool (__fastcall *v16)(__int64); // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 (*v19)(void); // rax
  __int64 v21; // [rsp-10h] [rbp-60h]
  int v22; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v23; // [rsp+10h] [rbp-40h] BYREF
  __int64 v24; // [rsp+18h] [rbp-38h]

  v4 = &a2[a3];
  v5 = a2;
  if ( v4 == a2 )
    return 1;
  while ( 1 )
  {
    v17 = *v5;
    if ( **(_BYTE **)a1 == 3 )
    {
      sub_250D230(&v23, v17, 2, 0);
    }
    else
    {
      v24 = 0;
      v23 = v17 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
    }
    v18 = v23;
    v6 = sub_257C550(*(_QWORD *)(a1 + 16), v23, v24, *(_QWORD *)(a1 + 24), 0, 0, 1);
    v7 = v6;
    if ( !v6 )
      return 0;
    v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    v9 = v8 == sub_2534FF0 ? v7 + 88 : ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v8)(v7, v18, v21);
    v10 = *(_DWORD **)(a1 + 40);
    v11 = *(unsigned int *)(v9 + 12);
    v12 = v10[3];
    v13 = *(void (__fastcall **)(__int64, int))(*(_QWORD *)v10 + 48LL);
    if ( v13 == sub_25352F0 )
    {
      v14 = v10[2] | v12 & v11;
      v10[3] = v14;
    }
    else
    {
      v22 = v10[3];
      v13((__int64)v10, v11);
      v14 = v10[3];
      v12 = v22;
    }
    sub_250C0C0(*(int **)(a1 + 32), v12 == v14);
    v15 = *(_DWORD **)(a1 + 40);
    v16 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v15 + 24LL);
    if ( v16 != sub_25350B0 )
      break;
    if ( v15[3] == v15[2] )
      goto LABEL_16;
LABEL_11:
    if ( v4 == ++v5 )
      return 1;
  }
  if ( !((unsigned __int8 (*)(void))v16)() )
    goto LABEL_11;
  v15 = *(_DWORD **)(a1 + 40);
LABEL_16:
  v19 = *(__int64 (**)(void))(*(_QWORD *)v15 + 16LL);
  if ( (char *)v19 == (char *)sub_2506010 )
    return v15[3] != 0;
  else
    return v19();
}
