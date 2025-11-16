// Function: sub_399E650
// Address: 0x399e650
//
__int64 *__fastcall sub_399E650(__int64 a1, const char *a2, _DWORD *a3, __int64 *a4, __int64 *a5, int **a6)
{
  int v10; // edx
  size_t v11; // rax
  char v12; // al
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  const char *v17; // r8
  size_t v18; // rdx
  int v19; // r9d
  __int64 v20; // r11
  __int64 v21; // r10
  unsigned int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rdi
  int *v25; // rax
  int v26; // edx
  int v28; // [rsp+4h] [rbp-5Ch]
  __int64 v29; // [rsp+8h] [rbp-58h]
  const char *v30; // [rsp+10h] [rbp-50h]
  size_t v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]

  *(_QWORD *)a1 = &unk_49EED30;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A3DF28;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_4A3DED8;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = &unk_4A3DEB8;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = a1;
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  v11 = strlen(a2);
  sub_16B8280(a1, a2, v11);
  v12 = *a3 & 3;
  v13 = *a5;
  v14 = *a4;
  *(_BYTE *)(a1 + 12) = (32 * v12) | *(_BYTE *)(a1 + 12) & 0x9F;
  v15 = a4[1];
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 48) = v15;
  v16 = v13 + 40LL * *((unsigned int *)a5 + 2);
  while ( v16 != v13 )
  {
    v17 = *(const char **)v13;
    v18 = *(_QWORD *)(v13 + 8);
    v19 = *(_DWORD *)(v13 + 16);
    v20 = *(_QWORD *)(v13 + 24);
    v21 = *(_QWORD *)(v13 + 32);
    v22 = *(_DWORD *)(a1 + 208);
    if ( v22 >= *(_DWORD *)(a1 + 212) )
    {
      v28 = *(_DWORD *)(v13 + 16);
      v29 = *(_QWORD *)(v13 + 24);
      v30 = *(const char **)v13;
      v31 = *(_QWORD *)(v13 + 8);
      v32 = *(_QWORD *)(v13 + 32);
      sub_399E260(a1 + 200, 0);
      v19 = v28;
      v20 = v29;
      v22 = *(_DWORD *)(a1 + 208);
      v17 = v30;
      v18 = v31;
      v21 = v32;
    }
    v23 = *(_QWORD *)(a1 + 200) + 48LL * v22;
    if ( v23 )
    {
      *(_QWORD *)v23 = v17;
      *(_QWORD *)(v23 + 8) = v18;
      *(_QWORD *)(v23 + 16) = v20;
      *(_QWORD *)(v23 + 24) = v21;
      *(_DWORD *)(v23 + 40) = v19;
      *(_BYTE *)(v23 + 44) = 1;
      *(_QWORD *)(v23 + 32) = &unk_4A3DEB8;
      v22 = *(_DWORD *)(a1 + 208);
    }
    v24 = *(_QWORD *)(a1 + 192);
    v13 += 40;
    *(_DWORD *)(a1 + 208) = v22 + 1;
    sub_16B7FD0(v24, v17, v18);
  }
  v25 = *a6;
  v26 = **a6;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v26;
  *(_DWORD *)(a1 + 176) = *v25;
  return sub_16B88A0(a1);
}
