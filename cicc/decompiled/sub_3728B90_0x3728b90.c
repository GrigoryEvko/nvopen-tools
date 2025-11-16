// Function: sub_3728B90
// Address: 0x3728b90
//
__int64 __fastcall sub_3728B90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  void (*v5)(); // rax
  unsigned __int16 v6; // ax
  __int64 v7; // rdi
  void (*v8)(); // rax
  __int64 v9; // rdi
  void (*v10)(); // rax
  _QWORD v12[4]; // [rsp+30h] [rbp-50h] BYREF
  char v13; // [rsp+50h] [rbp-30h]
  char v14; // [rsp+51h] [rbp-2Fh]

  if ( !byte_5050C30 && (unsigned int)sub_2207590((__int64)&byte_5050C30) )
  {
    byte_5050C38 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL);
    sub_2207640((__int64)&byte_5050C30);
  }
  v14 = 1;
  v13 = 3;
  v12[0] = "Length of contribution";
  v2 = sub_31F0F50(a2);
  v3 = *(_QWORD *)(a2 + 224);
  v4 = v2;
  v5 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  v14 = 1;
  v12[0] = "DWARF version number";
  v13 = 3;
  if ( v5 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v5)(v3, v12, 1);
  v6 = sub_31DF670(a2);
  sub_31DC9F0(a2, v6);
  v7 = *(_QWORD *)(a2 + 224);
  v8 = *(void (**)())(*(_QWORD *)v7 + 120LL);
  v14 = 1;
  v12[0] = "Address size";
  v13 = 3;
  if ( v8 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v8)(v7, v12, 1);
  sub_31DC9D0(a2, (unsigned __int8)byte_5050C38);
  v9 = *(_QWORD *)(a2 + 224);
  v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v14 = 1;
  v12[0] = "Segment selector size";
  v13 = 3;
  if ( v10 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v10)(v9, v12, 1);
  sub_31DC9D0(a2, 0);
  return v4;
}
