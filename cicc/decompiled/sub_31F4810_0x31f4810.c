// Function: sub_31F4810
// Address: 0x31f4810
//
__int64 __fastcall sub_31F4810(__int64 a1)
{
  __int64 v2; // r13
  void (__fastcall *v3)(__int64, _QWORD, _QWORD); // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  void (*v6)(); // rax
  __int64 v7; // rdi
  void (*v8)(); // rax
  __int64 v9; // rdi
  void (*v10)(); // rax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 i; // rbx
  __int64 v15; // rsi
  __int64 *v16; // rdi
  __int64 v17; // rax
  unsigned __int8 (*v18)(void); // rdx
  __int64 v19; // rdi
  void (*v20)(); // rax
  int v21; // [rsp+2Ch] [rbp-114h]
  char *v22; // [rsp+30h] [rbp-110h] BYREF
  __int64 v23; // [rsp+38h] [rbp-108h]
  __int64 v24; // [rsp+40h] [rbp-100h]
  _BYTE v25[40]; // [rsp+48h] [rbp-F8h] BYREF
  _QWORD v26[8]; // [rsp+70h] [rbp-D0h] BYREF
  char *v27; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v28; // [rsp+B8h] [rbp-88h]
  _QWORD *v29; // [rsp+C0h] [rbp-80h]
  __int64 v30; // [rsp+C8h] [rbp-78h]
  __int16 v31; // [rsp+D0h] [rbp-70h]
  _QWORD v32[2]; // [rsp+D8h] [rbp-68h] BYREF
  void *v33; // [rsp+E8h] [rbp-58h] BYREF
  int v34; // [rsp+F0h] [rbp-50h]
  _QWORD v35[9]; // [rsp+F8h] [rbp-48h] BYREF

  v2 = *(_QWORD *)(a1 + 528);
  v3 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v2 + 176LL);
  v4 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v3(v2, *(_QWORD *)(v4 + 408), 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 528) + 608LL))(
    *(_QWORD *)(a1 + 528),
    2,
    0,
    1,
    0);
  v5 = *(_QWORD *)(a1 + 528);
  v6 = *(void (**)())(*(_QWORD *)v5 + 120LL);
  v27 = "Magic";
  v31 = 259;
  if ( v6 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v6)(v5, &v27, 1);
    v5 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 536LL))(v5, 20171205, 4);
  v7 = *(_QWORD *)(a1 + 528);
  v8 = *(void (**)())(*(_QWORD *)v7 + 120LL);
  v27 = "Section Version";
  v31 = 259;
  if ( v8 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v8)(v7, &v27, 1);
    v7 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v7 + 536LL))(v7, 0, 2);
  v9 = *(_QWORD *)(a1 + 528);
  v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v27 = "Hash Algorithm";
  v31 = 259;
  if ( v10 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v10)(v9, &v27, 1);
    v9 = *(_QWORD *)(a1 + 528);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v9 + 536LL))(v9, 2, 2);
  v21 = 4096;
  result = sub_3707D10(a1 + 632);
  v13 = result + 8 * v12;
  for ( i = result; v13 != i; result = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64))(v17 + 520))(v16, v15, 8) )
  {
    v16 = *(__int64 **)(a1 + 528);
    v17 = *v16;
    v18 = *(unsigned __int8 (**)(void))(*v16 + 96);
    if ( (char *)v18 != (char *)sub_C13EE0 )
    {
      if ( v18() )
      {
        v23 = 0;
        v22 = v25;
        v26[5] = 0x100000000LL;
        v24 = 32;
        v26[1] = 2;
        v26[0] = &unk_49DD288;
        memset(&v26[2], 0, 24);
        v26[6] = &v22;
        sub_CB5980((__int64)v26, 0, 0, 0);
        v30 = 2;
        v27 = "{0:X+} [{1}]";
        v29 = v35;
        v28 = 12;
        LOBYTE(v31) = 1;
        v32[0] = &unk_4A35470;
        v32[1] = i;
        v34 = v21;
        v33 = &unk_49E65E8;
        v35[0] = &v33;
        v35[1] = v32;
        sub_CB6840((__int64)v26, (__int64)&v27);
        v19 = *(_QWORD *)(a1 + 528);
        v20 = *(void (**)())(*(_QWORD *)v19 + 120LL);
        v31 = 261;
        v27 = v22;
        v28 = v23;
        if ( v20 != nullsub_98 )
          ((void (__fastcall *)(__int64, char **, __int64))v20)(v19, &v27, 1);
        ++v21;
        v26[0] = &unk_49DD388;
        sub_CB5840((__int64)v26);
        if ( v22 != v25 )
          _libc_free((unsigned __int64)v22);
      }
      v16 = *(__int64 **)(a1 + 528);
      v17 = *v16;
    }
    v15 = i;
    i += 8;
  }
  return result;
}
