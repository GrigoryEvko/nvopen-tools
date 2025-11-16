// Function: sub_2DEA620
// Address: 0x2dea620
//
__int64 __fastcall sub_2DEA620(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 (*v7)(); // rdx
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rcx
  size_t v12; // r8
  __int64 v13; // r9
  __int64 (*v14)(); // rdx
  int v15; // eax
  void *v16; // rsi
  __int64 v18; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+8h] [rbp-A8h]
  int v20; // [rsp+10h] [rbp-A0h]
  __int64 v21; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v22; // [rsp+28h] [rbp-88h]
  int v23; // [rsp+30h] [rbp-80h]
  int v24; // [rsp+34h] [rbp-7Ch]
  int v25; // [rsp+38h] [rbp-78h]
  char v26; // [rsp+3Ch] [rbp-74h]
  _QWORD v27[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v28; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v29; // [rsp+58h] [rbp-58h]
  __int64 v30; // [rsp+60h] [rbp-50h]
  int v31; // [rsp+68h] [rbp-48h]
  char v32; // [rsp+6Ch] [rbp-44h]
  _BYTE v33[64]; // [rsp+70h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v7 == sub_23CE270 )
    BUG();
  v8 = v6 + 8;
  v9 = ((__int64 (__fastcall *)(_QWORD, __int64))v7)(*a2, a3);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v10 == sub_2C8F680 )
  {
    v18 = v8;
    v19 = 0;
    BUG();
  }
  v18 = v8;
  v19 = ((__int64 (__fastcall *)(__int64))v10)(v9);
  v14 = *(__int64 (**)())(*(_QWORD *)v19 + 1504LL);
  v15 = 2;
  if ( v14 != sub_2DE6890 )
    v15 = ((__int64 (__fastcall *)(__int64))v14)(v19);
  v20 = v15;
  v16 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2DE7CB0((__int64)&v18, a3, (__int64)v14, v11, v12, v13) )
  {
    v22 = v27;
    v23 = 2;
    v27[0] = &unk_4F82408;
    v25 = 0;
    v26 = 1;
    v28 = 0;
    v29 = v33;
    v30 = 2;
    v31 = 0;
    v32 = 1;
    v24 = 1;
    v21 = 1;
    sub_C8CF70(a1, v16, 2, (__int64)v27, (__int64)&v21);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v33, (__int64)&v28);
    if ( !v32 )
      _libc_free((unsigned __int64)v29);
    if ( !v26 )
      _libc_free((unsigned __int64)v22);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v16;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
