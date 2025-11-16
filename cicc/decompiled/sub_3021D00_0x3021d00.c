// Function: sub_3021D00
// Address: 0x3021d00
//
void __fastcall sub_3021D00(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  int v5; // r12d
  unsigned int v6; // r15d
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // r14
  void (__fastcall *v13)(__int64, _QWORD, _QWORD); // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdx
  _QWORD *v18; // rdi
  void *v19; // rdx
  __int64 *v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  _QWORD v24[4]; // [rsp+0h] [rbp-2C0h] BYREF
  __int16 v25; // [rsp+20h] [rbp-2A0h]
  _QWORD v26[3]; // [rsp+30h] [rbp-290h] BYREF
  __int64 v27; // [rsp+48h] [rbp-278h]
  __int64 v28; // [rsp+50h] [rbp-270h]
  __int64 v29; // [rsp+58h] [rbp-268h]
  unsigned __int64 *v30; // [rsp+60h] [rbp-260h]
  unsigned __int64 v31[3]; // [rsp+70h] [rbp-250h] BYREF
  _BYTE v32[568]; // [rsp+88h] [rbp-238h] BYREF

  v3 = sub_BA8DC0(a2, (__int64)"llvm.ident", 10);
  if ( v3 )
  {
    v4 = v3;
    v5 = sub_B91A00(v3);
    if ( v5 )
    {
      v6 = 0;
      while ( 1 )
      {
        v10 = sub_B91A10(v4, v6);
        v11 = *(_BYTE *)(v10 - 16);
        v7 = (v11 & 2) != 0 ? *(__int64 **)(v10 - 32) : (__int64 *)(v10 - 8LL * ((v11 >> 2) & 0xF) - 16);
        v8 = sub_B91420(*v7);
        if ( v9 == 10 && *(_QWORD *)v8 == 0x6564692E6363766ELL && *(_WORD *)(v8 + 8) == 29806 )
          break;
        if ( v5 == ++v6 )
          return;
      }
      v29 = 0x100000000LL;
      v30 = v31;
      v26[0] = &unk_49DD288;
      v31[0] = (unsigned __int64)v32;
      v31[1] = 0;
      v31[2] = 512;
      v26[1] = 2;
      v26[2] = 0;
      v27 = 0;
      v28 = 0;
      sub_CB5980((__int64)v26, 0, 0, 0);
      v12 = *(_QWORD *)(a1 + 224);
      v13 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v12 + 176LL);
      v14 = sub_31DA6B0(a1);
      v13(v12, *(_QWORD *)(v14 + 904), 0);
      if ( (unsigned __int64)(v27 - v28) <= 1 )
      {
        sub_CB6200((__int64)v26, (unsigned __int8 *)"\t\"", 2u);
      }
      else
      {
        *(_WORD *)v28 = 8713;
        v28 += 2;
      }
      v15 = sub_904010((__int64)v26, (const char *)off_4C5D100);
      sub_904010(v15, "; ");
      v16 = sub_904010((__int64)v26, (const char *)off_4C5D0F8);
      sub_904010(v16, "; ");
      v17 = (_QWORD *)v28;
      if ( (unsigned __int64)(v27 - v28) <= 8 )
      {
        v23 = sub_CB6200((__int64)v26, (unsigned __int8 *)"Based on ", 9u);
        v19 = *(void **)(v23 + 32);
        v18 = (_QWORD *)v23;
      }
      else
      {
        *(_BYTE *)(v28 + 8) = 32;
        v18 = v26;
        *v17 = 0x6E6F206465736142LL;
        v19 = (void *)(v28 + 9);
        v28 += 9;
      }
      if ( v18[3] - (_QWORD)v19 <= 0xAu )
      {
        v18 = (_QWORD *)sub_CB6200((__int64)v18, "NVVM 20.0.0", 0xBu);
      }
      else
      {
        qmemcpy(v19, "NVVM 20.0.0", 11);
        v18[4] += 11LL;
      }
      sub_904010((__int64)v18, "\"\n");
      v20 = *(__int64 **)(a1 + 224);
      v21 = v30[1];
      v22 = *v30;
      v25 = 261;
      v24[0] = v22;
      v24[1] = v21;
      sub_E99A90(v20, (__int64)v24);
      v26[0] = &unk_49DD388;
      sub_CB5840((__int64)v26);
      if ( (_BYTE *)v31[0] != v32 )
        _libc_free(v31[0]);
    }
  }
}
