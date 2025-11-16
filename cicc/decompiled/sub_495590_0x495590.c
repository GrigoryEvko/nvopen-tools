// Function: sub_495590
// Address: 0x495590
//
__int64 __fastcall sub_495590(__int64 a1, const char *a2, __int64 *a3, __int64 *a4, _BYTE *a5, _QWORD *a6, _BYTE *a7)
{
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  size_t v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v17; // rax
  __int64 v20; // [rsp+10h] [rbp-70h]
  const char *v22; // [rsp+20h] [rbp-60h] BYREF
  char v23; // [rsp+40h] [rbp-40h]
  char v24; // [rsp+41h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v10 = sub_C57470();
  v11 = *(unsigned int *)(a1 + 80);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v20 = v10;
    sub_C8D5F0(a1 + 72, a1 + 88, v11 + 1, 8);
    v11 = *(unsigned int *)(a1 + 80);
    v10 = v20;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v11) = v10;
  *(_QWORD *)(a1 + 144) = off_49E1308;
  *(_QWORD *)a1 = &off_49E1328;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = &unk_49DC350;
  *(_QWORD *)(a1 + 184) = nullsub_319;
  *(_QWORD *)(a1 + 176) = sub_E3E8E0;
  v12 = strlen(a2);
  sub_C53080(a1, a2, v12);
  v13 = *a3;
  *(_QWORD *)(a1 + 64) = a3[1];
  v14 = a4[1];
  *(_QWORD *)(a1 + 56) = v13;
  v15 = *a4;
  *(_QWORD *)(a1 + 48) = v14;
  *(_QWORD *)(a1 + 40) = v15;
  v16 = *(_QWORD *)(a1 + 136) == 0;
  *(_BYTE *)(a1 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  if ( v16 )
  {
    *(_QWORD *)(a1 + 136) = *a6;
  }
  else
  {
    v17 = sub_CEADF0();
    v24 = 1;
    v22 = "cl::location(x) specified more than once!";
    v23 = 3;
    sub_C53280(a1, &v22, 0, 0, v17);
  }
  *(_BYTE *)(a1 + 12) = (8 * (*a7 & 3)) | *(_BYTE *)(a1 + 12) & 0xE7;
  return sub_C53130(a1);
}
