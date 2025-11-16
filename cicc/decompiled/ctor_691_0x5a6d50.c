// Function: ctor_691
// Address: 0x5a6d50
//
int __fastcall ctor_691(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  __int64 v7; // r9
  __int64 v8; // r9
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v13; // [rsp+8h] [rbp-48h] BYREF
  const char *v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+18h] [rbp-38h]

  v13 = &v12;
  v14 = "The number of searches per loop in the window algorithm. 0 means no search number limit.";
  v15 = 88;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7340)(
    &unk_5040740,
    "window-search-num",
    &v14,
    (char *)&v12 + 4,
    &v13,
    a6,
    0x100000006LL);
  __cxa_atexit(sub_984970, &unk_5040740, &qword_4A427C0);
  v13 = &v12;
  v14 = "The ratio of searches per loop in the window algorithm. 100 means search all positions in the loop, while 0 mean"
        "s not performing any search.";
  v15 = 140;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7560)(
    &unk_5040660,
    "window-search-ratio",
    &v14,
    (char *)&v12 + 4,
    &v13,
    v6,
    0x100000028LL);
  __cxa_atexit(sub_984970, &unk_5040660, &qword_4A427C0);
  v13 = &v12;
  v14 = "The coefficient used when initializing II in the window algorithm.";
  v15 = 66;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7780)(
    &unk_5040580,
    "window-ii-coeff",
    &v14,
    (char *)&v12 + 4,
    &v13,
    v7,
    0x100000005LL);
  __cxa_atexit(sub_984970, &unk_5040580, &qword_4A427C0);
  v13 = &v12;
  v14 = "The lower limit of the scheduling region in the window algorithm.";
  v15 = 65;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7560)(
    &unk_50404A0,
    "window-region-limit",
    &v14,
    (char *)&v12 + 4,
    &v13,
    v8,
    0x100000003LL);
  __cxa_atexit(sub_984970, &unk_50404A0, &qword_4A427C0);
  v13 = &v12;
  v14 = "The lower limit of the difference between best II and base II in the window algorithm. If the difference is smal"
        "ler than this lower limit, window scheduling will not be performed.";
  v15 = 179;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7340)(
    &unk_50403C0,
    "window-diff-limit",
    &v14,
    (char *)&v12 + 4,
    &v13,
    v9,
    0x100000002LL);
  __cxa_atexit(sub_984970, &unk_50403C0, &qword_4A427C0);
  v13 = &v12;
  v14 = "The upper limit of II in the window algorithm.";
  v15 = 46;
  ((void (__fastcall *)(void *, const char *, const char **, char *, __int64 **, __int64, __int64))sub_35E7780)(
    &unk_50402E0,
    "window-ii-limit",
    &v14,
    (char *)&v12 + 4,
    &v13,
    v10,
    0x1000003E8LL);
  return __cxa_atexit(sub_984970, &unk_50402E0, &qword_4A427C0);
}
