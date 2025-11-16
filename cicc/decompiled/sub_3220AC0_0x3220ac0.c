// Function: sub_3220AC0
// Address: 0x3220ac0
//
void __fastcall sub_3220AC0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // rdi
  void (*v7)(); // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  void (__fastcall **v12[4])(_QWORD, _QWORD, _QWORD *); // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v7 = *(void (**)())(*(_QWORD *)v6 + 120LL);
  v14 = 1;
  v12[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD *))"Loc expr size";
  v13 = 3;
  if ( v7 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD, __int64))v7)(v6, v12, 1);
  if ( (unsigned __int16)sub_3220AA0(a1) > 4u )
  {
    if ( (((__int64)a2 - *(_QWORD *)(a1 + 1432)) >> 5) + 1 == *(_DWORD *)(a1 + 1440) )
      v8 = *(_QWORD *)(a1 + 2480);
    else
      v8 = a2[6];
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 424LL))(
      *(_QWORD *)(a1 + 8),
      v8 - a2[2],
      0,
      0);
    goto LABEL_10;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = a2[2];
  if ( (((__int64)a2 - *(_QWORD *)(a1 + 1432)) >> 5) + 1 == *(_DWORD *)(a1 + 1440) )
  {
    v11 = *(_QWORD *)(a1 + 2480) - v10;
    if ( v11 <= 0xFFFF )
      goto LABEL_9;
  }
  else
  {
    v11 = a2[6] - v10;
    if ( v11 <= 0xFFFF )
    {
LABEL_9:
      sub_31DC9F0(v9, v11);
LABEL_10:
      v12[1] = *(void (__fastcall ***)(_QWORD, _QWORD, _QWORD *))(a1 + 8);
      v12[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD *))&unk_4A35708;
      sub_321FA30(a1, v12, a2, a3);
      return;
    }
  }
  sub_31DC9F0(v9, 0);
}
