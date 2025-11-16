// Function: ctor_683_0
// Address: 0x5a5b20
//
int __fastcall ctor_683_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned int v8; // eax
  char *v9; // rbx
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  char v14; // [rsp+Ch] [rbp-184h] BYREF
  char v15; // [rsp+Dh] [rbp-183h] BYREF
  char v16; // [rsp+Eh] [rbp-182h] BYREF
  char v17; // [rsp+Fh] [rbp-181h] BYREF
  _BYTE v18[32]; // [rsp+10h] [rbp-180h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-160h] BYREF
  _BYTE v20[32]; // [rsp+50h] [rbp-140h] BYREF
  _QWORD v21[10]; // [rsp+70h] [rbp-120h] BYREF
  _BYTE v22[80]; // [rsp+C0h] [rbp-D0h] BYREF
  _BYTE v23[80]; // [rsp+110h] [rbp-80h] BYREF
  char v24; // [rsp+160h] [rbp-30h] BYREF

  qword_503FB00 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_503FB0C &= 0x8000u;
  word_503FB10 = 0;
  qword_503FB50 = 0x100000000LL;
  qword_503FB18 = 0;
  qword_503FB20 = 0;
  qword_503FB28 = 0;
  dword_503FB08 = v4;
  qword_503FB30 = 0;
  qword_503FB38 = 0;
  qword_503FB40 = 0;
  qword_503FB48 = (__int64)&unk_503FB58;
  qword_503FB60 = 0;
  qword_503FB68 = (__int64)&unk_503FB80;
  qword_503FB70 = 1;
  dword_503FB78 = 0;
  byte_503FB7C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503FB50;
  v7 = (unsigned int)qword_503FB50 + 1LL;
  if ( v7 > HIDWORD(qword_503FB50) )
  {
    sub_C8D5F0((char *)&unk_503FB58 - 16, &unk_503FB58, v7, 8);
    v6 = (unsigned int)qword_503FB50;
  }
  *(_QWORD *)(qword_503FB48 + 8 * v6) = v5;
  qword_503FB88 = (__int64)&byte_503FB98;
  qword_503FBB0 = (__int64)&byte_503FBC0;
  LODWORD(qword_503FB50) = qword_503FB50 + 1;
  qword_503FB90 = 0;
  qword_503FBA8 = (__int64)&unk_49DC130;
  byte_503FB98 = 0;
  byte_503FBC0 = 0;
  qword_503FB00 = (__int64)&unk_49DC010;
  qword_503FBB8 = 0;
  byte_503FBD0 = 0;
  qword_503FBD8 = (__int64)&unk_49DC350;
  qword_503FBF8 = (__int64)nullsub_92;
  qword_503FBF0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_503FB00, "regalloc-priority-interactive-channel-base", 42);
  qword_503FB30 = 215;
  LOBYTE(dword_503FB0C) = dword_503FB0C & 0x9F | 0x20;
  qword_503FB28 = (__int64)"Base file path for the interactive mode. The incoming filename should have the name <regalloc"
                           "-priority-interactive-channel-base>.in, while the outgoing name should be <regalloc-priority-"
                           "interactive-channel-base>.out";
  sub_C53130(&qword_503FB00);
  __cxa_atexit(sub_BC5A40, &qword_503FB00, &qword_4A427C0);
  v21[0] = 1;
  sub_3595F90(&unk_503FAD0, v21, 1, v20);
  __cxa_atexit(sub_3593040, &unk_503FAD0, &qword_4A427C0);
  v19[0] = 1;
  sub_3595F90(v20, v19, 1, v18);
  sub_A758C0(v21, "priority", &v17);
  v8 = sub_310D000();
  sub_310F6F0(&unk_503FA80, v21, 0, v8, 4, v20);
  sub_2240A30(v21);
  sub_30FC6B0(v20);
  __cxa_atexit(sub_30FB2C0, &unk_503FA80, &qword_4A427C0);
  v9 = &v24;
  sub_A758C0(v20, "li_size", &v17);
  v10 = sub_310D010();
  sub_310F6F0(v21, v20, 0, v10, 8, &unk_503FAD0);
  sub_A758C0(v19, "stage", &v16);
  v11 = sub_310D010();
  sub_310F6F0(v22, v19, 0, v11, 8, &unk_503FAD0);
  sub_A758C0(v18, "weight", &v15);
  v12 = sub_310D000();
  sub_310F6F0(v23, v18, 0, v12, 4, &unk_503FAD0);
  sub_30FC510(&unk_503FA60, v21, 3, &v14);
  do
  {
    v9 -= 80;
    sub_30FC6B0(v9 + 40);
    sub_2240A30(v9);
  }
  while ( v9 != (char *)v21 );
  sub_2240A30(v18);
  sub_2240A30(v19);
  sub_2240A30(v20);
  return __cxa_atexit(sub_30FB110, &unk_503FA60, &qword_4A427C0);
}
