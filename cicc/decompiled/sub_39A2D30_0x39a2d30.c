// Function: sub_39A2D30
// Address: 0x39a2d30
//
__int64 __fastcall sub_39A2D30(__int64 a1, char a2, unsigned __int8 a3)
{
  __int64 v5; // r12
  __int64 v6; // rdi
  void (*v7)(); // rax
  int v8; // eax
  __int64 v9; // rdi
  void (*v10)(); // rax
  unsigned int v11; // r12d
  __int64 v12; // rdi
  __int64 v13; // r8
  void (*v14)(); // rax
  __int64 v15; // rdi
  __int64 v16; // r8
  void (*v17)(); // rax
  __int64 v18; // rdi
  __int64 v19; // r8
  void (*v20)(); // rax
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // rdi
  __int64 v24; // r8
  void (*v25)(); // rax
  _QWORD v26[2]; // [rsp+0h] [rbp-50h] BYREF
  char v27; // [rsp+10h] [rbp-40h]
  char v28; // [rsp+11h] [rbp-3Fh]

  v5 = *(_QWORD *)(a1 + 192);
  v6 = *(_QWORD *)(v5 + 256);
  v7 = *(void (**)())(*(_QWORD *)v6 + 104LL);
  v28 = 1;
  v26[0] = "Length of Unit";
  v27 = 3;
  if ( v7 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v7)(v6, v26, 1);
    v5 = *(_QWORD *)(a1 + 192);
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  sub_396F340(v5, *(_DWORD *)(a1 + 28) + v8);
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 256LL);
  v10 = *(void (**)())(*(_QWORD *)v9 + 104LL);
  v28 = 1;
  v26[0] = "DWARF version number";
  v27 = 3;
  if ( v10 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v10)(v9, v26, 1);
  v11 = (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200));
  sub_396F320(*(_QWORD *)(a1 + 192), v11);
  if ( v11 > 4 )
  {
    v12 = *(_QWORD *)(a1 + 192);
    v13 = *(_QWORD *)(v12 + 256);
    v14 = *(void (**)())(*(_QWORD *)v13 + 104LL);
    v28 = 1;
    v26[0] = "DWARF Unit Type";
    v27 = 3;
    if ( v14 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v14)(v13, v26, 1);
      v12 = *(_QWORD *)(a1 + 192);
    }
    sub_396F300(v12, a3);
    v15 = *(_QWORD *)(a1 + 192);
    v16 = *(_QWORD *)(v15 + 256);
    v17 = *(void (**)())(*(_QWORD *)v16 + 104LL);
    v28 = 1;
    v26[0] = "Address Size (in bytes)";
    v27 = 3;
    if ( v17 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v16, v26, 1);
      v15 = *(_QWORD *)(a1 + 192);
    }
    sub_396F300(v15, *(_DWORD *)(*(_QWORD *)(v15 + 240) + 8LL));
  }
  v18 = *(_QWORD *)(a1 + 192);
  v19 = *(_QWORD *)(v18 + 256);
  v20 = *(void (**)())(*(_QWORD *)v19 + 104LL);
  v28 = 1;
  v26[0] = "Offset Into Abbrev. Section";
  v27 = 3;
  if ( v20 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v20)(v19, v26, 1);
    v18 = *(_QWORD *)(a1 + 192);
  }
  v21 = sub_396DD80(v18);
  if ( a2 )
  {
    result = sub_396F340(*(_QWORD *)(a1 + 192), 0);
    if ( v11 > 4 )
      return result;
  }
  else
  {
    result = (__int64)sub_397C410(*(_QWORD *)(a1 + 192), *(_QWORD *)(*(_QWORD *)(v21 + 80) + 8LL), 0);
    if ( v11 > 4 )
      return result;
  }
  v23 = *(_QWORD *)(a1 + 192);
  v24 = *(_QWORD *)(v23 + 256);
  v25 = *(void (**)())(*(_QWORD *)v24 + 104LL);
  v28 = 1;
  v26[0] = "Address Size (in bytes)";
  v27 = 3;
  if ( v25 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v24, v26, 1);
    v23 = *(_QWORD *)(a1 + 192);
  }
  return sub_396F300(v23, *(_DWORD *)(*(_QWORD *)(v23 + 240) + 8LL));
}
