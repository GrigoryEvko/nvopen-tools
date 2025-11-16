// Function: sub_2FF2920
// Address: 0x2ff2920
//
__int64 __fastcall sub_2FF2920(_BYTE *a1)
{
  int v2; // ecx
  __int64 v3; // rax
  char v4; // r12
  __int64 (*v5)(); // rdx
  __int64 result; // rax
  __int64 v7; // rax
  char v8; // r13
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  void (*v11)(); // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  void (*v14)(); // rdx
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  void (*v17)(); // rdx
  __int64 (*v18)(); // rax
  __int64 v19; // rsi
  bool (__fastcall *v20)(__int64); // rax
  bool v21; // di
  _QWORD *v22; // rsi
  __int64 v23[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v24[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = qword_5028508;
  *(_BYTE *)(*((_QWORD *)a1 + 32) + 688LL) = (2 * ((_DWORD)qword_5028508 != 2))
                                           | *(_BYTE *)(*((_QWORD *)a1 + 32) + 688LL) & 0xFD;
  if ( v2 == 1 )
  {
    v3 = *((_QWORD *)a1 + 32);
    goto LABEL_15;
  }
  v3 = *((_QWORD *)a1 + 32);
  if ( (_DWORD)qword_5028428 != 1 && ((*(_BYTE *)(v3 + 865) & 0x10) == 0 || (_DWORD)qword_5028428 == 2) )
  {
    if ( *(_DWORD *)(v3 + 648) || (*(_BYTE *)(v3 + 688) & 2) == 0 )
    {
      v4 = a1[251];
      a1[251] = 0;
      goto LABEL_8;
    }
LABEL_15:
    *(_BYTE *)(v3 + 865) |= 8u;
    *(_BYTE *)(*((_QWORD *)a1 + 32) + 865LL) &= ~0x10u;
    v4 = a1[251];
    a1[251] = 0;
    goto LABEL_8;
  }
  *(_BYTE *)(v3 + 865) &= ~8u;
  *(_BYTE *)(*((_QWORD *)a1 + 32) + 865LL) |= 0x10u;
  v4 = a1[251];
  if ( !sub_2FF2900((__int64)a1) )
    a1[251] = 0;
  v7 = *(_QWORD *)a1;
  v8 = a1[250];
  a1[250] = 1;
  v9 = *(__int64 (**)())(v7 + 192);
  if ( v9 == sub_2FEDA40 || ((unsigned __int8 (__fastcall *)(_BYTE *))v9)(a1) )
    goto LABEL_13;
  v10 = *(_QWORD *)a1;
  v11 = *(void (**)())(*(_QWORD *)a1 + 200LL);
  if ( v11 != nullsub_1689 )
  {
    ((void (__fastcall *)(_BYTE *))v11)(a1);
    v10 = *(_QWORD *)a1;
  }
  v12 = *(__int64 (**)())(v10 + 208);
  if ( v12 == sub_2FEDA60 || ((unsigned __int8 (__fastcall *)(_BYTE *))v12)(a1) )
    goto LABEL_13;
  v13 = *(_QWORD *)a1;
  v14 = *(void (**)())(*(_QWORD *)a1 + 216LL);
  if ( v14 != nullsub_1690 )
  {
    ((void (__fastcall *)(_BYTE *))v14)(a1);
    v13 = *(_QWORD *)a1;
  }
  v15 = *(__int64 (**)())(v13 + 224);
  if ( v15 == sub_2FEDA80 || ((unsigned __int8 (__fastcall *)(_BYTE *))v15)(a1) )
    goto LABEL_13;
  v16 = *(_QWORD *)a1;
  v17 = *(void (**)())(*(_QWORD *)a1 + 232LL);
  if ( v17 != nullsub_1691 )
  {
    ((void (__fastcall *)(_BYTE *))v17)(a1);
    v16 = *(_QWORD *)a1;
  }
  v18 = *(__int64 (**)())(v16 + 240);
  if ( v18 == sub_2FEDAA0 || ((unsigned __int8 (__fastcall *)(_BYTE *))v18)(a1) )
  {
LABEL_13:
    a1[250] = v8;
    result = 1;
    goto LABEL_9;
  }
  a1[250] = v8;
  v19 = sub_2FF2900((__int64)a1);
  v20 = *(bool (__fastcall **)(__int64))(*(_QWORD *)a1 + 256LL);
  if ( v20 == sub_2FEDE00 )
  {
    v21 = *(_DWORD *)(*((_QWORD *)a1 + 32) + 868LL) == 2;
  }
  else
  {
    v19 = (unsigned int)v19;
    v21 = v20((__int64)a1);
  }
  v22 = (_QWORD *)sub_35C8A90(v21, v19);
  sub_2FF0E80((__int64)a1, v22, 1u);
  if ( !sub_2FF2900((__int64)a1) )
  {
LABEL_8:
    v5 = *(__int64 (**)())(*(_QWORD *)a1 + 184LL);
    result = 1;
    if ( v5 == sub_2FEDA30 )
      goto LABEL_9;
    result = ((__int64 (__fastcall *)(_BYTE *))v5)(a1);
    if ( (_BYTE)result )
      goto LABEL_9;
  }
  sub_2FF12A0((__int64)a1, &unk_501D67C, 1u);
  v23[0] = (__int64)v24;
  sub_2FEE630(v23, "After Instruction Selection", (__int64)"");
  sub_2FF0D00((__int64)a1, (__int64)v23);
  if ( (_QWORD *)v23[0] != v24 )
    j_j___libc_free_0(v23[0]);
  result = 0;
LABEL_9:
  a1[251] = v4;
  return result;
}
