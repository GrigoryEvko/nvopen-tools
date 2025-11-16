// Function: sub_2586F10
// Address: 0x2586f10
//
__int64 __fastcall sub_2586F10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edi
  _BYTE *v9; // r13
  signed int v10; // r14d
  __int64 v11; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  void (__fastcall *v20)(__int64, unsigned __int64, unsigned __int64); // rax
  _QWORD *v21; // rdi
  __int64 (*v22)(void); // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // [rsp+0h] [rbp-80h] BYREF
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+10h] [rbp-70h]
  char *v29; // [rsp+18h] [rbp-68h] BYREF
  __int64 v30; // [rsp+20h] [rbp-60h]
  char v31; // [rsp+28h] [rbp-58h] BYREF
  __int64 v32; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v33; // [rsp+38h] [rbp-48h] BYREF
  __int64 v34; // [rsp+40h] [rbp-40h]
  _BYTE v35[56]; // [rsp+48h] [rbp-38h] BYREF

  v7 = *a2;
  v8 = *((_DWORD *)a2 + 4);
  v29 = &v31;
  v30 = 0;
  v28 = v7;
  if ( v8 )
  {
    v9 = v35;
    sub_2538240((__int64)&v29, (char **)a2 + 1, a3, a4, a5, a6);
    v10 = **(_DWORD **)a1;
    v33 = v35;
    v34 = 0;
    v32 = v28;
    if ( (_DWORD)v30 )
      sub_2538550((__int64)&v33, (__int64)&v29, v23, (unsigned int)v30, v24, v25);
  }
  else
  {
    v9 = v35;
    v10 = **(_DWORD **)a1;
    v32 = v7;
    v33 = v35;
    v34 = 0;
  }
  v26 = sub_254CA10((__int64)&v32, v10);
  v27 = v11;
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  if ( (unsigned __int8)sub_2509800(&v26)
    && (v13 = v26, (v14 = sub_2584D90(*(_QWORD *)(a1 + 8), v26, v27, *(_QWORD *)(a1 + 16), 0, 0, 1)) != 0) )
  {
    v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 48LL);
    if ( v15 == sub_2534F40 )
      v16 = v14 + 88;
    else
      v16 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v14, v13);
    v17 = *(_QWORD *)(a1 + 24);
    if ( !*(_BYTE *)(v17 + 24) )
    {
      *(_QWORD *)(v17 + 8) = 1;
      *(_QWORD *)(v17 + 16) = 0x100000000LL;
      *(_BYTE *)(v17 + 24) = 1;
      *(_QWORD *)v17 = &unk_4A16ED8;
      v17 = *(_QWORD *)(a1 + 24);
    }
    v18 = *(_QWORD *)(v16 + 8);
    v19 = *(_QWORD *)(v16 + 16);
    v20 = *(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64))(*(_QWORD *)v17 + 72LL);
    if ( v20 == sub_2535490 )
    {
      if ( *(_QWORD *)(v17 + 8) <= v18 )
        v18 = *(_QWORD *)(v17 + 8);
      if ( *(_QWORD *)(v17 + 16) <= v19 )
        v19 = *(_QWORD *)(v17 + 16);
      *(_QWORD *)(v17 + 8) = v18;
      *(_QWORD *)(v17 + 16) = v19;
    }
    else
    {
      v20(v17, v19, v18);
    }
    v21 = *(_QWORD **)(a1 + 24);
    v22 = *(__int64 (**)(void))(*v21 + 16LL);
    if ( (char *)v22 == (char *)sub_2505FD0 )
      LOBYTE(v9) = v21[2] != 1;
    else
      LODWORD(v9) = v22();
  }
  else
  {
    LODWORD(v9) = 0;
  }
  if ( v29 != &v31 )
    _libc_free((unsigned __int64)v29);
  return (unsigned int)v9;
}
