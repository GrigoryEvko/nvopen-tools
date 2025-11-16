// Function: sub_31DE9E0
// Address: 0x31de9e0
//
__int64 __fastcall sub_31DE9E0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 (*v6)(); // rax
  char v7; // al
  int v8; // edx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // r13
  char v12; // al
  int v13; // edx
  int v14; // ecx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // rbx
  unsigned __int8 *v21; // rax
  __int64 (*v22)(); // rbx
  _BYTE *v23; // rax
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r9
  __int64 (*v29)(); // rax
  __int64 v30; // rbx
  __int64 (*v31)(); // rax
  __int64 result; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-B0h]
  void (__fastcall *v39)(__int64, _QWORD, __int64, _QWORD); // [rsp+8h] [rbp-A8h]
  void (__fastcall *v40)(__int64, _QWORD, __int64, _QWORD); // [rsp+8h] [rbp-A8h]
  __int64 v41; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v42; // [rsp+10h] [rbp-A0h]

  v5 = *(_DWORD *)(*(_QWORD *)(a1 + 200) + 564LL);
  if ( v5 == 3 )
  {
    v35 = sub_31DB510(a1, a3);
    if ( (*(_BYTE *)(a3 + 32) & 0xF) != 0 && *(_QWORD *)(*(_QWORD *)(a1 + 208) + 304LL) )
    {
      if ( (((*(_BYTE *)(a3 + 32) & 0xF) + 14) & 0xFu) <= 3 )
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
          *(_QWORD *)(a1 + 224),
          v35,
          26);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
        *(_QWORD *)(a1 + 224),
        v35,
        9);
    }
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), v35, 3);
    sub_31DE970(a1, v35, (*(_BYTE *)(a3 + 32) >> 4) & 3, 1);
    v36 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 240LL))(a1, *(_QWORD *)(a3 - 32));
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 272LL))(
      *(_QWORD *)(a1 + 224),
      v35,
      v36);
    result = sub_31DE680(a1, a3, v37);
    if ( v35 != result )
      return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 272LL))(
               *(_QWORD *)(a1 + 224),
               result,
               v36);
  }
  else
  {
    if ( v5 != 5 || (v6 = *(__int64 (**)())(*(_QWORD *)a1 + 392LL), v6 == sub_3020030) || !v6() )
      sub_C64ED0("IFuncs are not supported on this platform", 1u);
    v7 = (unsigned __int8)sub_BD5D20(a3);
    v11 = sub_31DE8D0(a1, a2, v8, (unsigned int)".lazy_pointer", v9, v10, v7);
    v12 = (unsigned __int8)sub_BD5D20(a3);
    v17 = sub_31DE8D0(a1, 773, v13, v14, v15, v16, v12);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 216) + 168LL) + 32LL),
      0);
    LODWORD(v18) = sub_AE4380(a2 + 312, 0);
    v19 = -1;
    if ( (_DWORD)v18 )
    {
      _BitScanReverse64((unsigned __int64 *)&v18, (unsigned int)v18);
      v19 = 63 - (v18 ^ 0x3F);
    }
    sub_31DCA70(a1, v19, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v11, 0);
    sub_31DE970(a1, v11, (*(_BYTE *)(a3 + 32) >> 4) & 3, 1);
    v20 = *(_QWORD *)(a1 + 224);
    v21 = (unsigned __int8 *)sub_E808D0(v17, 0, *(_QWORD **)(a1 + 216), 0);
    sub_E9A5B0(v20, v21);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 216) + 168LL) + 24LL),
      0);
    v41 = *(_QWORD *)(a1 + 200);
    v22 = *(__int64 (**)())(*(_QWORD *)v41 + 16LL);
    v23 = sub_B30850(a3);
    if ( v22 == sub_23CE270 )
      BUG();
    v24 = ((__int64 (__fastcall *)(__int64, _BYTE *))v22)(v41, v23);
    v25 = *(__int64 (**)())(*(_QWORD *)v24 + 144LL);
    if ( v25 == sub_2C8F680 )
      BUG();
    v26 = a3;
    v42 = *(_BYTE *)(((__int64 (__fastcall *)(__int64))v25)(v24) + 74);
    v27 = sub_31DB510(a1, a3);
    if ( (*(_BYTE *)(a3 + 32) & 0xF) != 0 && *(_QWORD *)(*(_QWORD *)(a1 + 208) + 304LL) )
    {
      if ( (((*(_BYTE *)(a3 + 32) & 0xF) + 14) & 0xFu) <= 3 )
      {
        v26 = v27;
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
          *(_QWORD *)(a1 + 224),
          v27,
          26);
      }
    }
    else
    {
      v26 = v27;
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
        *(_QWORD *)(a1 + 224),
        v27,
        9);
    }
    v28 = *(_QWORD *)(a1 + 224);
    v29 = *(__int64 (**)())(*(_QWORD *)a1 + 392LL);
    if ( v29 == sub_3020030 )
    {
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v28 + 616LL))(v28, v42, 0, 0);
    }
    else
    {
      v38 = *(_QWORD *)(a1 + 224);
      v40 = *(void (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v28 + 616LL);
      v34 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v29)(a1, v26, 0);
      v40(v38, v42, v34, 0);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v27, 0);
    sub_31DE970(a1, v27, (*(_BYTE *)(a3 + 32) >> 4) & 3, 1);
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 400LL))(a1, a2, a3, v11);
    v30 = *(_QWORD *)(a1 + 224);
    v31 = *(__int64 (**)())(*(_QWORD *)a1 + 392LL);
    if ( v31 == sub_3020030 )
    {
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v30 + 616LL))(v30, v42, 0, 0);
    }
    else
    {
      v39 = *(void (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v30 + 616LL);
      v33 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v31)(a1, a2, 0);
      v39(v30, v42, v33, 0);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v17, 0);
    sub_31DE970(a1, v17, (*(_BYTE *)(a3 + 32) >> 4) & 3, 1);
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 408LL))(a1, a2, a3, v11);
  }
  return result;
}
