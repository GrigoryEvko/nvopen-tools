// Function: sub_3597390
// Address: 0x3597390
//
void __fastcall __noreturn sub_3597390(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  char *v9; // r13
  size_t v10; // r14
  char *v11; // rax
  int v12; // eax
  void *v13; // rdx
  void *v14; // rax
  __int64 v15; // [rsp+8h] [rbp-108h] BYREF
  _QWORD v16[4]; // [rsp+10h] [rbp-100h] BYREF
  __int64 v17[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v18[4]; // [rsp+50h] [rbp-C0h] BYREF
  void *v19[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v20; // [rsp+90h] [rbp-80h]
  _BYTE v21[112]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = a3[1] - *a3;
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 16) = 1;
  *(_QWORD *)a1 = &unk_4A399F0;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 4) + 1;
  if ( v7 <= 0xFFFFFFFFFFFFFFFLL )
  {
    *(_QWORD *)(a1 + 24) = 0;
    v9 = 0;
    *(_QWORD *)(a1 + 32) = 0;
    v10 = 8 * v7;
    *(_QWORD *)(a1 + 40) = 0;
    if ( v7 )
    {
      v11 = (char *)sub_22077B0(8 * v7);
      v9 = &v11[v10];
      *(_QWORD *)(a1 + 24) = v11;
      *(_QWORD *)(a1 + 40) = &v11[v10];
      if ( v11 != &v11[v10] )
        memset(v11, 0, v10);
    }
    *(_QWORD *)(a1 + 32) = v9;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)a1 = &unk_4A39A50;
    *(_DWORD *)(a1 + 72) = -1;
    *(_QWORD *)(a1 + 80) = sub_22077B0(1u);
    v15 = 2;
    sub_3595F90(v16, &v15, 1);
    sub_35929F0(v17, "model_selector");
    v12 = sub_310D020();
    sub_310F6F0((__int64)v21, (__int64)v17, 0, v12, 8, (__int64)v16);
    v13 = *(void **)a6;
    v14 = *(void **)(a6 + 8);
    v19[2] = v21;
    v19[0] = v13;
    v19[1] = v14;
    v20 = 1029;
    sub_CA0F50(v18, v19);
    BUG();
  }
  sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
}
