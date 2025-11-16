// Function: sub_1ED8620
// Address: 0x1ed8620
//
void __fastcall sub_1ED8620(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdi
  __int64 (*v8)(); // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  _BYTE *v11; // [rsp+0h] [rbp-130h] BYREF
  __int64 v12; // [rsp+8h] [rbp-128h]
  _BYTE v13[32]; // [rsp+10h] [rbp-120h] BYREF
  _QWORD v14[3]; // [rsp+30h] [rbp-100h] BYREF
  __int64 v15; // [rsp+48h] [rbp-E8h]
  __int64 v16; // [rsp+50h] [rbp-E0h]
  __int64 v17; // [rsp+58h] [rbp-D8h]
  __int64 v18; // [rsp+60h] [rbp-D0h]
  __int64 v19; // [rsp+68h] [rbp-C8h]
  int v20; // [rsp+70h] [rbp-C0h]
  char v21; // [rsp+74h] [rbp-BCh]
  __int64 v22; // [rsp+78h] [rbp-B8h]
  __int64 v23; // [rsp+80h] [rbp-B0h]
  _BYTE *v24; // [rsp+88h] [rbp-A8h]
  _BYTE *v25; // [rsp+90h] [rbp-A0h]
  __int64 v26; // [rsp+98h] [rbp-98h]
  int v27; // [rsp+A0h] [rbp-90h]
  _BYTE v28[32]; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v29; // [rsp+C8h] [rbp-68h]
  _BYTE *v30; // [rsp+D0h] [rbp-60h]
  _BYTE *v31; // [rsp+D8h] [rbp-58h]
  __int64 v32; // [rsp+E0h] [rbp-50h]
  int v33; // [rsp+E8h] [rbp-48h]
  _BYTE v34[64]; // [rsp+F0h] [rbp-40h] BYREF

  v1 = a1 + 232;
  v3 = *(_QWORD *)(a1 + 240);
  v14[2] = &v11;
  v12 = 0x800000000LL;
  v4 = *(_QWORD *)(a1 + 272);
  v5 = *(_QWORD *)(v3 + 40);
  v17 = 0;
  v16 = v4;
  v6 = 0;
  v15 = v5;
  v7 = *(_QWORD *)(v3 + 16);
  v14[0] = &unk_4A00C10;
  v11 = v13;
  v14[1] = 0;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 40LL);
  v9 = 0;
  if ( v8 != sub_1D00B00 )
  {
    v10 = ((__int64 (__fastcall *)(__int64, __int64 (*)(), _QWORD, _QWORD))v8)(v7, v8, 0, 0);
    v6 = v12;
    v9 = v10;
    v5 = v15;
  }
  v20 = v6;
  v18 = v9;
  v24 = v28;
  v25 = v28;
  v30 = v34;
  v31 = v34;
  v19 = v1;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v26 = 4;
  v27 = 0;
  v29 = 0;
  v32 = 4;
  v33 = 0;
  *(_QWORD *)(v5 + 8) = v14;
  sub_21020C0(v14, a1 + 664, 0, 0, 0);
  *(_QWORD *)(v15 + 8) = 0;
  if ( v31 != v30 )
    _libc_free((unsigned __int64)v31);
  if ( v25 != v24 )
    _libc_free((unsigned __int64)v25);
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
}
