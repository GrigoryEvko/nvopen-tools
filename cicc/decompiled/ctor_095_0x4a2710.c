// Function: ctor_095
// Address: 0x4a2710
//
int ctor_095()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-58h]
  int v13; // [rsp+10h] [rbp-50h] BYREF
  int v14; // [rsp+14h] [rbp-4Ch] BYREF
  int *v15; // [rsp+18h] [rbp-48h] BYREF
  const char *v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]

  qword_4F91380 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F913D0 = 0x100000000LL;
  dword_4F9138C &= 0x8000u;
  word_4F91390 = 0;
  qword_4F91398 = 0;
  qword_4F913A0 = 0;
  dword_4F91388 = v0;
  qword_4F913A8 = 0;
  qword_4F913B0 = 0;
  qword_4F913B8 = 0;
  qword_4F913C0 = 0;
  qword_4F913C8 = (__int64)&unk_4F913D8;
  qword_4F913E0 = 0;
  qword_4F913E8 = (__int64)&unk_4F91400;
  qword_4F913F0 = 1;
  dword_4F913F8 = 0;
  byte_4F913FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F913D0;
  v3 = (unsigned int)qword_4F913D0 + 1LL;
  if ( v3 > HIDWORD(qword_4F913D0) )
  {
    sub_C8D5F0((char *)&unk_4F913D8 - 16, &unk_4F913D8, v3, 8);
    v2 = (unsigned int)qword_4F913D0;
  }
  *(_QWORD *)(qword_4F913C8 + 8 * v2) = v1;
  qword_4F91410 = (__int64)&unk_49D9748;
  qword_4F91380 = (__int64)&unk_49DC090;
  qword_4F91420 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F913D0) = qword_4F913D0 + 1;
  qword_4F91440 = (__int64)nullsub_23;
  qword_4F91408 = 0;
  qword_4F91438 = (__int64)sub_984030;
  qword_4F91418 = 0;
  sub_C53080(&qword_4F91380, "enable-double-float-shrink", 26);
  LOWORD(qword_4F91418) = 256;
  LOBYTE(qword_4F91408) = 0;
  qword_4F913B0 = 58;
  LOBYTE(dword_4F9138C) = dword_4F9138C & 0x9F | 0x20;
  qword_4F913A8 = (__int64)"Enable unsafe double to float shrinking for math lib calls";
  sub_C53130(&qword_4F91380);
  __cxa_atexit(sub_984900, &qword_4F91380, &qword_4A427C0);
  qword_4F912A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F912F0 = 0x100000000LL;
  dword_4F912AC &= 0x8000u;
  qword_4F912E8 = (__int64)&unk_4F912F8;
  word_4F912B0 = 0;
  qword_4F912B8 = 0;
  dword_4F912A8 = v4;
  qword_4F912C0 = 0;
  qword_4F912C8 = 0;
  qword_4F912D0 = 0;
  qword_4F912D8 = 0;
  qword_4F912E0 = 0;
  qword_4F91300 = 0;
  qword_4F91308 = (__int64)&unk_4F91320;
  qword_4F91310 = 1;
  dword_4F91318 = 0;
  byte_4F9131C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F912F0;
  if ( (unsigned __int64)(unsigned int)qword_4F912F0 + 1 > HIDWORD(qword_4F912F0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F912F8 - 16, &unk_4F912F8, (unsigned int)qword_4F912F0 + 1LL, 8);
    v6 = (unsigned int)qword_4F912F0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F912E8 + 8 * v6) = v5;
  qword_4F91330 = (__int64)&unk_49D9748;
  qword_4F912A0 = (__int64)&unk_49DC090;
  qword_4F91340 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F912F0) = qword_4F912F0 + 1;
  qword_4F91360 = (__int64)nullsub_23;
  qword_4F91328 = 0;
  qword_4F91358 = (__int64)sub_984030;
  qword_4F91338 = 0;
  sub_C53080(&qword_4F912A0, "optimize-hot-cold-new", 21);
  LOWORD(qword_4F91338) = 256;
  LOBYTE(qword_4F91328) = 0;
  qword_4F912D0 = 42;
  LOBYTE(dword_4F912AC) = dword_4F912AC & 0x9F | 0x20;
  qword_4F912C8 = (__int64)"Enable hot/cold operator new library calls";
  sub_C53130(&qword_4F912A0);
  __cxa_atexit(sub_984900, &qword_4F912A0, &qword_4A427C0);
  qword_4F911C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F91210 = 0x100000000LL;
  dword_4F911CC &= 0x8000u;
  word_4F911D0 = 0;
  qword_4F91208 = (__int64)&unk_4F91218;
  qword_4F911D8 = 0;
  dword_4F911C8 = v7;
  qword_4F911E0 = 0;
  qword_4F911E8 = 0;
  qword_4F911F0 = 0;
  qword_4F911F8 = 0;
  qword_4F91200 = 0;
  qword_4F91220 = 0;
  qword_4F91228 = (__int64)&unk_4F91240;
  qword_4F91230 = 1;
  dword_4F91238 = 0;
  byte_4F9123C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F91210;
  v10 = (unsigned int)qword_4F91210 + 1LL;
  if ( v10 > HIDWORD(qword_4F91210) )
  {
    sub_C8D5F0((char *)&unk_4F91218 - 16, &unk_4F91218, v10, 8);
    v9 = (unsigned int)qword_4F91210;
  }
  *(_QWORD *)(qword_4F91208 + 8 * v9) = v8;
  qword_4F91250 = (__int64)&unk_49D9748;
  qword_4F911C0 = (__int64)&unk_49DC090;
  qword_4F91260 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4F91210) = qword_4F91210 + 1;
  qword_4F91280 = (__int64)nullsub_23;
  qword_4F91248 = 0;
  qword_4F91278 = (__int64)sub_984030;
  qword_4F91258 = 0;
  sub_C53080(&qword_4F911C0, "optimize-existing-hot-cold-new", 30);
  LOBYTE(qword_4F91248) = 0;
  qword_4F911F0 = 67;
  LOBYTE(dword_4F911CC) = dword_4F911CC & 0x9F | 0x20;
  LOWORD(qword_4F91258) = 256;
  qword_4F911E8 = (__int64)"Enable optimization of existing hot/cold operator new library calls";
  sub_C53130(&qword_4F911C0);
  __cxa_atexit(sub_984900, &qword_4F911C0, &qword_4A427C0);
  v15 = &v13;
  v16 = "Value to pass to hot/cold operator new for cold allocation";
  v17 = 58;
  v13 = 1;
  v14 = 1;
  sub_4A2500((__int64)&unk_4F910E0, "cold-new-hint-value", &v14, &v15, (__int64 *)&v16);
  __cxa_atexit(sub_11DA1F0, &unk_4F910E0, &qword_4A427C0);
  v15 = &v13;
  v16 = "Value to pass to hot/cold operator new for notcold (warm) allocation";
  v17 = 68;
  v13 = 128;
  v14 = 1;
  sub_4A2500((__int64)&unk_4F91000, "notcold-new-hint-value", &v14, &v15, (__int64 *)&v16);
  __cxa_atexit(sub_11DA1F0, &unk_4F91000, &qword_4A427C0);
  v15 = &v13;
  v16 = "Value to pass to hot/cold operator new for hot allocation";
  v17 = 57;
  v13 = 254;
  v14 = 1;
  sub_4A2500((__int64)&unk_4F90F20, "hot-new-hint-value", &v14, &v15, (__int64 *)&v16);
  return __cxa_atexit(sub_11DA1F0, &unk_4F90F20, &qword_4A427C0);
}
