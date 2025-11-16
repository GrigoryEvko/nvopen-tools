// Function: sub_2599600
// Address: 0x2599600
//
char __fastcall sub_2599600(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v4; // r12
  unsigned __int64 *v5; // rbx
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdi
  _BYTE *v9; // r15
  __int64 v10; // rsi
  char v11; // al
  void (__fastcall *v12)(__int64, char); // rdx
  char v13; // si
  _BYTE *v14; // rdi
  bool (__fastcall *v15)(__int64); // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 (*v18)(void); // rax
  char v20; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]

  v4 = &a2[a3];
  v5 = a2;
  if ( v4 == a2 )
    return 1;
  while ( 1 )
  {
    v16 = *v5;
    if ( **(_BYTE **)a1 == 3 )
    {
      sub_250D230(&v21, v16, 2, 0);
    }
    else
    {
      v22 = 0;
      v21 = v16 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
    }
    v17 = v21;
    v6 = sub_25294B0(*(_QWORD *)(a1 + 16), v21, v22, *(_QWORD *)(a1 + 24), 0, 0, 1);
    if ( !v6 )
      return 0;
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    v8 = v7 == sub_2534F20 ? v6 + 88 : ((__int64 (__fastcall *)(__int64, unsigned __int64))v7)(v6, v17);
    v9 = *(_BYTE **)(a1 + 40);
    v10 = *(unsigned __int8 *)(v8 + 9);
    v11 = v9[9];
    v12 = *(void (__fastcall **)(__int64, char))(*(_QWORD *)v9 + 48LL);
    if ( v12 == sub_25353A0 )
    {
      v13 = v9[8] | v11 & v10;
      v9[9] = v13;
    }
    else
    {
      v20 = v9[9];
      v12((__int64)v9, v10);
      v13 = v9[9];
      v11 = v20;
    }
    sub_250C0C0(*(int **)(a1 + 32), v13 == v11);
    v14 = *(_BYTE **)(a1 + 40);
    v15 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v14 + 24LL);
    if ( v15 != sub_2534F80 )
      break;
    if ( v14[9] == v14[8] )
      goto LABEL_16;
LABEL_11:
    if ( v4 == ++v5 )
      return 1;
  }
  if ( !((unsigned __int8 (*)(void))v15)() )
    goto LABEL_11;
  v14 = *(_BYTE **)(a1 + 40);
LABEL_16:
  v18 = *(__int64 (**)(void))(*(_QWORD *)v14 + 16LL);
  if ( (char *)v18 == (char *)sub_2505EE0 )
    return v14[9] != 0;
  else
    return v18();
}
