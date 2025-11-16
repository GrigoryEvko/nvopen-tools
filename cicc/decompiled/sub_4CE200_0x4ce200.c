// Function: sub_4CE200
// Address: 0x4ce200
//
__int64 __fastcall sub_4CE200(
        __int64 a1,
        const char *a2,
        __int64 *a3,
        __int64 *a4,
        _DWORD *a5,
        _QWORD *a6,
        _BYTE *a7,
        _BYTE *a8)
{
  int v10; // edx
  size_t v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rax
  const char *v20; // [rsp+20h] [rbp-50h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49EED30;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = &unk_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_QWORD *)(a1 + 168) = off_49EE380;
  *(_QWORD *)a1 = &off_49EE3A0;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 176) = &unk_49EEE90;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  v11 = strlen(a2);
  sub_16B8280(a1, a2, v11);
  v12 = *a3;
  *(_QWORD *)(a1 + 64) = a3[1];
  v13 = a4[1];
  *(_QWORD *)(a1 + 56) = v12;
  v14 = *a4;
  *(_QWORD *)(a1 + 48) = v13;
  LODWORD(v13) = *a5;
  *(_QWORD *)(a1 + 40) = v14;
  v15 = *(_QWORD *)(a1 + 160) == 0;
  *(_BYTE *)(a1 + 12) = (32 * (v13 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  if ( v15 )
  {
    *(_QWORD *)(a1 + 160) = *a6;
  }
  else
  {
    v16 = sub_16E8CB0();
    v22 = 1;
    v20 = "cl::location(x) specified more than once!";
    v21 = 3;
    sub_16B1F90(a1, &v20, 0, 0, v16);
  }
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a1 + 12) & 0xE0 | *a8 & 7 | (8 * (*a7 & 3));
  return sub_16B88A0(a1);
}
