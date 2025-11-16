// Function: sub_4CEA00
// Address: 0x4cea00
//
__int64 __fastcall sub_4CEA00(
        __int64 a1,
        const char *a2,
        __int64 *a3,
        _QWORD *a4,
        _BYTE *a5,
        _BYTE *a6,
        _QWORD *a7,
        __int64 *a8)
{
  size_t v10; // rax
  bool v11; // zf
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  const char *v19; // [rsp+20h] [rbp-50h] BYREF
  char v20; // [rsp+30h] [rbp-40h]
  char v21; // [rsp+31h] [rbp-3Fh]

  sub_12F0CE0(a1, 0, 0);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = off_49EEF30;
  *(_QWORD *)a1 = off_49EEF50;
  *(_BYTE *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 176) = &unk_49EEDB0;
  *(_QWORD *)(a1 + 184) = off_4985000;
  v10 = strlen(a2);
  sub_16B8280(a1, a2, v10);
  v11 = *(_QWORD *)(a1 + 160) == 0;
  v12 = *a3;
  *(_QWORD *)(a1 + 48) = a3[1];
  *(_QWORD *)(a1 + 40) = v12;
  if ( v11 )
  {
    *(_QWORD *)(a1 + 160) = *a4;
  }
  else
  {
    v13 = sub_16E8CB0();
    v21 = 1;
    v19 = "cl::location(x) specified more than once!";
    v20 = 3;
    sub_16B1F90(a1, &v19, 0, 0, v13);
  }
  v14 = *a8;
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a1 + 12) & 0x87 | (8 * (*a6 & 3)) | (32 * (*a5 & 3));
  *(_QWORD *)(a1 + 72) = *a7;
  sub_4CE9E0(v14, a1);
  return sub_16B88A0(a1);
}
