// Function: sub_3248790
// Address: 0x3248790
//
__int64 __fastcall sub_3248790(__int64 *a1, char a2, unsigned __int8 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rdi
  void (*v8)(); // rax
  unsigned int v9; // ebx
  __int64 v10; // rdi
  __int64 v11; // r8
  void (*v12)(); // rax
  __int64 v13; // rdi
  __int64 v14; // r8
  void (*v15)(); // rax
  __int64 v16; // rdi
  __int64 v17; // r8
  void (*v18)(); // rax
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21; // rdi
  __int64 v22; // r8
  void (*v23)(); // rax
  _QWORD v25[4]; // [rsp+40h] [rbp-60h] BYREF
  char v26; // [rsp+60h] [rbp-40h]
  char v27; // [rsp+61h] [rbp-3Fh]

  v4 = a1[23];
  v5 = *a1;
  v6 = *(_BYTE *)(a1[26] + 3689) == 0;
  v27 = 1;
  v26 = 3;
  v25[0] = "Length of Unit";
  if ( v6 )
  {
    (*(__int64 (**)(void))(v5 + 96))();
    a1[25] = sub_31F0F50(v4);
  }
  else
  {
    (*(void (**)(void))(v5 + 56))();
    sub_31F0F40(v4);
  }
  v7 = *(_QWORD *)(a1[23] + 224);
  v8 = *(void (**)())(*(_QWORD *)v7 + 120LL);
  v27 = 1;
  v25[0] = "DWARF version number";
  v26 = 3;
  if ( v8 != nullsub_98 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v8)(v7, v25, 1);
  v9 = (unsigned __int16)sub_3220AA0(a1[26]);
  sub_31DC9F0(a1[23], v9);
  if ( v9 > 4 )
  {
    v10 = a1[23];
    v11 = *(_QWORD *)(v10 + 224);
    v12 = *(void (**)())(*(_QWORD *)v11 + 120LL);
    v27 = 1;
    v25[0] = "DWARF Unit Type";
    v26 = 3;
    if ( v12 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v12)(v11, v25, 1);
      v10 = a1[23];
    }
    sub_31DC9D0(v10, a3);
    v13 = a1[23];
    v14 = *(_QWORD *)(v13 + 224);
    v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
    v27 = 1;
    v25[0] = "Address Size (in bytes)";
    v26 = 3;
    if ( v15 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v14, v25, 1);
      v13 = a1[23];
    }
    sub_31DC9D0(v13, *(_DWORD *)(*(_QWORD *)(v13 + 208) + 8LL));
  }
  v16 = a1[23];
  v17 = *(_QWORD *)(v16 + 224);
  v18 = *(void (**)())(*(_QWORD *)v17 + 120LL);
  v27 = 1;
  v25[0] = "Offset Into Abbrev. Section";
  v26 = 3;
  if ( v18 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v18)(v17, v25, 1);
    v16 = a1[23];
  }
  v19 = sub_31DA6B0(v16);
  if ( a2 )
  {
    result = sub_31F0F00(a1[23], 0);
    if ( v9 > 4 )
      return result;
  }
  else
  {
    result = (__int64)sub_31F0D70(a1[23], *(_QWORD *)(*(_QWORD *)(v19 + 80) + 16LL), 0);
    if ( v9 > 4 )
      return result;
  }
  v21 = a1[23];
  v22 = *(_QWORD *)(v21 + 224);
  v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
  v27 = 1;
  v25[0] = "Address Size (in bytes)";
  v26 = 3;
  if ( v23 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v22, v25, 1);
    v21 = a1[23];
  }
  return sub_31DC9D0(v21, *(_DWORD *)(*(_QWORD *)(v21 + 208) + 8LL));
}
