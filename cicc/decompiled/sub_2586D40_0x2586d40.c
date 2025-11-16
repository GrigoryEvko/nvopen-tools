// Function: sub_2586D40
// Address: 0x2586d40
//
char __fastcall sub_2586D40(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v4; // r12
  unsigned __int64 *v5; // rbx
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdi
  _QWORD *v9; // r15
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 (__fastcall *v12)(__int64, unsigned __int64); // rdx
  _QWORD *v13; // rdi
  bool (__fastcall *v14)(__int64); // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rsi
  __int64 (*v17)(void); // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v4 = &a2[a3];
  v5 = a2;
  if ( v4 == a2 )
    return 1;
  while ( 1 )
  {
    v15 = *v5;
    if ( **(_BYTE **)a1 == 3 )
    {
      sub_250D230(&v20, v15, 2, 0);
    }
    else
    {
      v21 = 0;
      v20 = v15 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
    }
    v16 = v20;
    v6 = sub_2584D90(*(_QWORD *)(a1 + 16), v20, v21, *(_QWORD *)(a1 + 24), 0, 0, 1);
    if ( !v6 )
      return 0;
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    v8 = v7 == sub_2534F40 ? v6 + 88 : ((__int64 (__fastcall *)(__int64, unsigned __int64))v7)(v6, v16);
    v9 = *(_QWORD **)(a1 + 40);
    v10 = *(_QWORD *)(v8 + 16);
    v11 = v9[2];
    v12 = *(unsigned __int64 (__fastcall **)(__int64, unsigned __int64))(*v9 + 48LL);
    if ( v12 == sub_2535430 )
    {
      if ( v11 <= v10 )
        v10 = v9[2];
      if ( v9[1] >= v10 )
        v10 = v9[1];
      v9[2] = v10;
    }
    else
    {
      v19 = v9[2];
      v12((__int64)v9, v10);
      v10 = v9[2];
      v11 = v19;
    }
    sub_250C0C0(*(int **)(a1 + 32), v10 == v11);
    v13 = *(_QWORD **)(a1 + 40);
    v14 = *(bool (__fastcall **)(__int64))(*v13 + 24LL);
    if ( v14 != sub_2535040 )
      break;
    if ( v13[2] == v13[1] )
      goto LABEL_20;
LABEL_15:
    if ( v4 == ++v5 )
      return 1;
  }
  if ( !((unsigned __int8 (*)(void))v14)() )
    goto LABEL_15;
  v13 = *(_QWORD **)(a1 + 40);
LABEL_20:
  v17 = *(__int64 (**)(void))(*v13 + 16LL);
  if ( (char *)v17 == (char *)sub_2505FD0 )
    return v13[2] != 1;
  else
    return v17();
}
