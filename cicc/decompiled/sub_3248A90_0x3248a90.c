// Function: sub_3248A90
// Address: 0x3248a90
//
__int64 __fastcall sub_3248A90(__int64 *a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v6; // dl
  __int64 v7; // rdi
  void (*v8)(); // rax
  __int64 v9; // rdi
  __int64 v10; // r8
  void (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17[4]; // [rsp+0h] [rbp-40h] BYREF
  char v18; // [rsp+20h] [rbp-20h]
  char v19; // [rsp+21h] [rbp-1Fh]

  if ( *(_BYTE *)(a1[26] + 3769)
    || (v15 = a1[23],
        v19 = 1,
        v17[0] = (__int64)"tu_begin",
        v18 = 3,
        v16 = sub_31DCC50(v15, v17, a3, a4, a5),
        a1[24] = v16,
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[23] + 224) + 208LL))(
          *(_QWORD *)(a1[23] + 224),
          v16,
          0),
        *(_BYTE *)(a1[26] + 3769)) )
  {
    v6 = 6;
  }
  else
  {
    v6 = 2;
  }
  sub_3248790(a1, a2, v6);
  v7 = *(_QWORD *)(a1[23] + 224);
  v8 = *(void (**)())(*(_QWORD *)v7 + 120LL);
  v19 = 1;
  v17[0] = (__int64)"Type Signature";
  v18 = 3;
  if ( v8 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int64 *, __int64))v8)(v7, v17, 1);
    v7 = *(_QWORD *)(a1[23] + 224);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v7 + 536LL))(v7, a1[49], 8);
  v9 = a1[23];
  v10 = *(_QWORD *)(v9 + 224);
  v11 = *(void (**)())(*(_QWORD *)v10 + 120LL);
  v19 = 1;
  v17[0] = (__int64)"Type DIE Offset";
  v18 = 3;
  if ( v11 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int64 *, __int64))v11)(v10, v17, 1);
    v9 = a1[23];
  }
  v12 = a1[50];
  v13 = 0;
  if ( v12 )
    v13 = *(unsigned int *)(v12 + 16);
  return sub_31F0F00(v9, v13);
}
