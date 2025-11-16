// Function: ctor_694
// Address: 0x5a77b0
//
int __fastcall ctor_694(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v15; // [rsp+4h] [rbp-3Ch] BYREF
  const char *v16; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v17[6]; // [rsp+10h] [rbp-30h] BYREF

  v16 = byte_3F871B3;
  v17[0] = "Override unique ID of ctor/dtor globals.";
  v15 = 1;
  v17[1] = 40;
  qword_5040CC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5040D10 = 0x100000000LL;
  word_5040CD0 = 0;
  dword_5040CCC &= 0x8000u;
  qword_5040CD8 = 0;
  qword_5040CE0 = 0;
  dword_5040CC8 = v4;
  qword_5040CE8 = 0;
  qword_5040CF0 = 0;
  qword_5040CF8 = 0;
  qword_5040D00 = 0;
  qword_5040D08 = (__int64)&unk_5040D18;
  qword_5040D20 = 0;
  qword_5040D28 = (__int64)&unk_5040D40;
  qword_5040D30 = 1;
  dword_5040D38 = 0;
  byte_5040D3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040D10;
  v7 = (unsigned int)qword_5040D10 + 1LL;
  if ( v7 > HIDWORD(qword_5040D10) )
  {
    sub_C8D5F0((char *)&unk_5040D18 - 16, &unk_5040D18, v7, 8);
    v6 = (unsigned int)qword_5040D10;
  }
  *(_QWORD *)(qword_5040D08 + 8 * v6) = v5;
  qword_5040D48 = (__int64)&byte_5040D58;
  qword_5040D70 = (__int64)&byte_5040D80;
  LODWORD(qword_5040D10) = qword_5040D10 + 1;
  qword_5040D50 = 0;
  qword_5040D68 = (__int64)&unk_49DC130;
  byte_5040D58 = 0;
  byte_5040D80 = 0;
  qword_5040CC0 = (__int64)&unk_49DC010;
  qword_5040D78 = 0;
  byte_5040D90 = 0;
  qword_5040D98 = (__int64)&unk_49DC350;
  qword_5040DB8 = (__int64)nullsub_92;
  qword_5040DB0 = (__int64)sub_BC4D70;
  sub_36D42C0(&qword_5040CC0, "nvptx-lower-global-ctor-dtor-id", v17, &v16, &v15);
  sub_C53130(&qword_5040CC0);
  __cxa_atexit(sub_BC5A40, &qword_5040CC0, &qword_4A427C0);
  qword_5040BE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_5040CC0, v8, v9), 1u);
  byte_5040C5C = 1;
  qword_5040C30 = 0x100000000LL;
  dword_5040BEC &= 0x8000u;
  qword_5040BF8 = 0;
  qword_5040C00 = 0;
  qword_5040C08 = 0;
  dword_5040BE8 = v10;
  word_5040BF0 = 0;
  qword_5040C10 = 0;
  qword_5040C18 = 0;
  qword_5040C20 = 0;
  qword_5040C28 = (__int64)&unk_5040C38;
  qword_5040C40 = 0;
  qword_5040C48 = (__int64)&unk_5040C60;
  qword_5040C50 = 1;
  dword_5040C58 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5040C30;
  v13 = (unsigned int)qword_5040C30 + 1LL;
  if ( v13 > HIDWORD(qword_5040C30) )
  {
    sub_C8D5F0((char *)&unk_5040C38 - 16, &unk_5040C38, v13, 8);
    v12 = (unsigned int)qword_5040C30;
  }
  *(_QWORD *)(qword_5040C28 + 8 * v12) = v11;
  LODWORD(qword_5040C30) = qword_5040C30 + 1;
  qword_5040C68 = 0;
  qword_5040C70 = (__int64)&unk_49D9748;
  qword_5040C78 = 0;
  qword_5040BE0 = (__int64)&unk_49DC090;
  qword_5040C80 = (__int64)&unk_49DC1D0;
  qword_5040CA0 = (__int64)nullsub_23;
  qword_5040C98 = (__int64)sub_984030;
  sub_C53080(&qword_5040BE0, "nvptx-emit-init-fini-kernel", 27);
  qword_5040C10 = 39;
  qword_5040C08 = (__int64)"Emit kernels to call ctor/dtor globals.";
  LOWORD(qword_5040C78) = 257;
  LOBYTE(qword_5040C68) = 1;
  LOBYTE(dword_5040BEC) = dword_5040BEC & 0x9F | 0x20;
  sub_C53130(&qword_5040BE0);
  return __cxa_atexit(sub_984900, &qword_5040BE0, &qword_4A427C0);
}
