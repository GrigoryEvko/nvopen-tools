// Function: sub_2E43BF0
// Address: 0x2e43bf0
//
void __fastcall sub_2E43BF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  size_t v8; // r13
  void *v9; // rax
  void *v10; // rdx
  size_t v11; // r13
  __int64 v12; // r12
  void (__fastcall *v13)(__int64, __int64); // rbx
  __int64 v14; // rax
  const void *v15; // r14
  const void *v16; // rax
  __int64 v17; // rdx
  const void *v18; // r14
  const void *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  void *v26[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v27; // [rsp+30h] [rbp-30h]

  v6 = *(_QWORD *)a1;
  if ( !v6 )
  {
    v24 = a4;
    v25 = a3;
    v21 = sub_22077B0(0xC0u);
    a3 = v25;
    a4 = v24;
    v6 = v21;
    if ( v21 )
    {
      *(_QWORD *)(v21 + 8) = 0;
      v22 = v21 + 32;
      *(_QWORD *)(v22 - 16) = 0;
      *(_QWORD *)(v22 - 8) = 0;
      *(_QWORD *)(v6 + 40) = v22;
      *(_QWORD *)(v6 + 32) = v22;
      *(_QWORD *)(v6 + 56) = v22;
      *(_QWORD *)(v6 + 96) = v6 + 88;
      *(_QWORD *)(v6 + 88) = v6 + 88;
      *(_QWORD *)(v6 + 48) = 0;
      *(_QWORD *)(v6 + 64) = 0;
      *(_QWORD *)(v6 + 72) = 0;
      *(_QWORD *)(v6 + 80) = 0;
      *(_QWORD *)v6 = &unk_4A289B8;
      *(_QWORD *)(v6 + 104) = 0;
      *(_QWORD *)(v6 + 112) = 0;
      *(_QWORD *)(v6 + 120) = 0;
      *(_QWORD *)(v6 + 128) = 0;
      *(_QWORD *)(v6 + 136) = 0;
      *(_QWORD *)(v6 + 144) = 0;
      *(_QWORD *)(v6 + 152) = 0;
      *(_QWORD *)(v6 + 160) = 0;
      *(_QWORD *)(v6 + 168) = 0;
      *(_QWORD *)(v6 + 176) = 0;
      *(_DWORD *)(v6 + 184) = 0;
    }
    v23 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v6;
    if ( v23 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
      v6 = *(_QWORD *)a1;
      a4 = v24;
      a3 = v25;
    }
  }
  sub_2E42FD0(v6, a2, a3, a4);
  v7 = (unsigned int)qword_501EFE8;
  if ( (_DWORD)qword_501EFE8 )
  {
    v8 = qword_4F8DF28[9];
    if ( !qword_4F8DF28[9]
      || (v6 = a2, v18 = (const void *)qword_4F8DF28[8], v19 = (const void *)sub_2E791E0(a2), v8 == v20)
      && (v7 = (__int64)v18, v6 = (__int64)v19, !memcmp(v19, v18, v8)) )
    {
      v9 = (void *)sub_2E791E0(a2);
      v6 = a1;
      v26[0] = "MachineBlockFrequencyDAGS.";
      v7 = (__int64)v26;
      v26[3] = v10;
      v27 = 1283;
      v26[2] = v9;
      sub_2E43B30(a1, v26, 1);
    }
  }
  if ( (_BYTE)qword_501ECA8 )
  {
    v11 = qword_4F8DA08[9];
    if ( !qword_4F8DA08[9]
      || (v15 = (const void *)qword_4F8DA08[8], v16 = (const void *)sub_2E791E0(a2), v11 == v17)
      && (v7 = (__int64)v15, v6 = (__int64)v16, !memcmp(v16, v15, v11)) )
    {
      v12 = *(_QWORD *)a1;
      v13 = *(void (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 24LL);
      v14 = sub_C5F790(v6, v7);
      v13(v12, v14);
    }
  }
}
