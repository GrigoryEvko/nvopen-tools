// Function: sub_25A1F90
// Address: 0x25a1f90
//
__int64 __fastcall sub_25A1F90(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned __int8 *v8; // rdi
  bool (__fastcall *v9)(__int64); // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 (*v13)(void); // rax
  __int64 v15; // [rsp+0h] [rbp-30h]

  v4 = &a2[a3];
  v5 = a2;
  if ( v4 == a2 )
    return 1;
  while ( 1 )
  {
    v10 = *v5 & 0xFFFFFFFFFFFFFFFCLL;
    if ( **(_BYTE **)a1 == 3 )
      v10 |= 1u;
    v15 = v10;
    nullsub_1518();
    v11 = sub_25803A0(*(_QWORD *)(a1 + 16), v15, 0, *(_QWORD *)(a1 + 24), 0, 0, 1);
    if ( !v11 )
      return 0;
    v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 48LL);
    v6 = v12 == sub_2534B10 ? v11 + 88 : ((__int64 (__fastcall *)(__int64, __int64))v12)(v11, v15);
    v7 = sub_25A1B90(*(_QWORD *)(a1 + 40), v6);
    sub_250C0C0(*(int **)(a1 + 32), v7);
    v8 = *(unsigned __int8 **)(a1 + 40);
    v9 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v8 + 24LL);
    if ( v9 != sub_2534FC0 )
      break;
    if ( v8[17] == v8[16] )
      goto LABEL_14;
LABEL_6:
    if ( v4 == ++v5 )
      return 1;
  }
  if ( !((unsigned __int8 (*)(void))v9)() )
    goto LABEL_6;
  v8 = *(unsigned __int8 **)(a1 + 40);
LABEL_14:
  v13 = *(__int64 (**)(void))(*(_QWORD *)v8 + 16LL);
  if ( (char *)v13 == (char *)sub_2505E40 )
    return v8[17];
  else
    return v13();
}
