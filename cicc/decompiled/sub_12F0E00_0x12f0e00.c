// Function: sub_12F0E00
// Address: 0x12f0e00
//
__int64 __fastcall sub_12F0E00(__int64 a1, const char *a2, __int64 *a3, __int64 *a4)
{
  int v6; // edx
  size_t v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+10h] [rbp-30h]
  char v18; // [rsp+11h] [rbp-2Fh]

  *(_QWORD *)a1 = &unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xF000 | 0x20;
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
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
  *(_QWORD *)a1 = &unk_49EEEB0;
  *(_QWORD *)(a1 + 160) = 0;
  v7 = strlen(a2);
  sub_16B8280(a1, a2, v7);
  v8 = a3[1];
  v9 = *a3;
  v10 = *(_QWORD *)(a1 + 160) == 0;
  v11 = *a4;
  *(_QWORD *)(a1 + 40) = v9;
  *(_QWORD *)(a1 + 48) = v8;
  if ( !v10 )
  {
    v12 = sub_16E8CB0(a1, a2, v9);
    a2 = (const char *)v16;
    v18 = 1;
    v16[0] = "cl::alias must only have one cl::aliasopt(...) specified!";
    v17 = 3;
    sub_16B1F90(a1, v16, 0, 0, v12);
  }
  v10 = *(_QWORD *)(a1 + 32) == 0;
  *(_QWORD *)(a1 + 160) = v11;
  if ( v10 )
  {
    v14 = sub_16E8CB0(a1, a2, v9);
    a2 = (const char *)v16;
    v18 = 1;
    v16[0] = "cl::alias must have argument name specified!";
    v17 = 3;
    sub_16B1F90(a1, v16, 0, 0, v14);
    v11 = *(_QWORD *)(a1 + 160);
    if ( v11 )
      goto LABEL_5;
  }
  else if ( v11 )
  {
    goto LABEL_5;
  }
  v15 = sub_16E8CB0(a1, a2, v9);
  v18 = 1;
  v16[0] = "cl::alias must have an cl::aliasopt(option) specified!";
  v17 = 3;
  sub_16B1F90(a1, v16, 0, 0, v15);
  v11 = *(_QWORD *)(a1 + 160);
LABEL_5:
  if ( v11 + 80 != a1 + 80 )
    sub_16CCD50();
  return sub_16B88A0(a1);
}
