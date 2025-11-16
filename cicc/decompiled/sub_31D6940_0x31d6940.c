// Function: sub_31D6940
// Address: 0x31d6940
//
void __fastcall sub_31D6940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 *v8; // r15
  unsigned __int64 v9; // r13
  const void *v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  void (__fastcall *v15)(__int64 *, __int64 *, _QWORD *, _QWORD, __int64, __int64, unsigned __int64, unsigned __int64, __int64); // rcx
  void (__fastcall *v16)(__int64 *, __int64 *, _QWORD *, __int64, __int64, __int64, unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-158h] BYREF
  unsigned __int64 v19; // [rsp+10h] [rbp-150h]
  unsigned __int64 v20; // [rsp+18h] [rbp-148h]
  __int64 v21; // [rsp+20h] [rbp-140h]
  _QWORD *v22; // [rsp+30h] [rbp-130h] BYREF
  __int64 v23; // [rsp+38h] [rbp-128h]
  _BYTE v24[16]; // [rsp+40h] [rbp-120h] BYREF
  _QWORD v25[8]; // [rsp+50h] [rbp-110h] BYREF
  unsigned __int64 v26[26]; // [rsp+90h] [rbp-D0h] BYREF

  v7 = *(_BYTE *)(a2 + 64) == 0;
  v8 = *(__int64 **)(a2 + 24);
  memset(v26, 0, 0xA0u);
  if ( v7 )
    goto LABEL_5;
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(const void **)(a2 + 32);
  v26[1] = 0;
  v26[0] = (unsigned __int64)&v26[3];
  v26[2] = 128;
  if ( v9 > 0x80 )
  {
    sub_C8D290((__int64)v26, &v26[3], v9, 1u, a5, a6);
    v17 = (unsigned __int64 *)(v26[0] + v26[1]);
LABEL_17:
    memcpy(v17, v10, v9);
    v9 += v26[1];
    goto LABEL_4;
  }
  if ( v9 )
  {
    v17 = &v26[3];
    goto LABEL_17;
  }
LABEL_4:
  v26[1] = v9;
  LOBYTE(v26[19]) = 1;
  sub_C849A0((__int64)v26);
LABEL_5:
  v25[5] = 0x100000000LL;
  v25[6] = &v22;
  v25[0] = &unk_49DD210;
  v22 = v24;
  v23 = 0;
  v24[0] = 0;
  memset(&v25[1], 0, 32);
  sub_CB5980((__int64)v25, 0, 0, 0);
  v14 = *v8;
  if ( LOBYTE(v26[19]) )
  {
    v15 = *(void (__fastcall **)(__int64 *, __int64 *, _QWORD *, _QWORD, __int64, __int64, unsigned __int64, unsigned __int64, __int64))(v14 + 24);
    LOBYTE(v21) = 1;
    v19 = v26[0];
    v20 = v26[1];
    v15(&v18, v8, v25, v15, v12, v13, v26[0], v26[1], v21);
  }
  else
  {
    v16 = *(void (__fastcall **)(__int64 *, __int64 *, _QWORD *, __int64, __int64, __int64, unsigned __int64, unsigned __int64, __int64))(v14 + 24);
    LOBYTE(v21) = 0;
    v16(&v18, v8, v25, v11, v12, v13, v19, v20, v21);
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 16LL))(v18);
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
    *(_QWORD *)(a1 + 224),
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 216) + 168LL) + 456LL),
    0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(a1 + 224) + 520LL))(*(_QWORD *)(a1 + 224), v22, v23);
  if ( v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  v25[0] = &unk_49DD210;
  sub_CB5840((__int64)v25);
  if ( v22 != (_QWORD *)v24 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( LOBYTE(v26[19]) )
  {
    if ( (unsigned __int64 *)v26[0] != &v26[3] )
      _libc_free(v26[0]);
  }
}
