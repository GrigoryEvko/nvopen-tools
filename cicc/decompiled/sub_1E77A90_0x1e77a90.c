// Function: sub_1E77A90
// Address: 0x1e77a90
//
_QWORD *__fastcall sub_1E77A90(__int64 a1, const char *a2, __int64 **a3, _DWORD *a4, __int64 *a5)
{
  int v8; // edx
  size_t v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // r11
  __int64 v16; // r10
  __int64 v17; // rcx
  const char *v18; // r9
  size_t v19; // r14
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdi
  const char *v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49FC720;
  *(_QWORD *)(a1 + 216) = a1 + 232;
  *(_QWORD *)(a1 + 224) = 0x800000000LL;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 192) = &unk_49FC698;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49FC678;
  *(_BYTE *)(a1 + 184) = 1;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 208) = a1;
  *(_QWORD *)(a1 + 200) = &unk_49FC6D0;
  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = *a3;
  v11 = **a3;
  *(_BYTE *)(a1 + 184) = 1;
  *(_QWORD *)(a1 + 160) = v11;
  *(_QWORD *)(a1 + 176) = *v10;
  v12 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v13 = a5[1];
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 48) = v13;
  sub_16B88A0(a1);
  v14 = (_QWORD *)qword_4FC7850[0];
  if ( qword_4FC7850[0] )
  {
    do
    {
      v15 = v14[3];
      v16 = v14[4];
      v17 = v14[5];
      v18 = (const char *)v14[1];
      v19 = v14[2];
      v20 = *(_DWORD *)(a1 + 224);
      if ( v20 >= *(_DWORD *)(a1 + 228) )
      {
        v24 = (const char *)v14[1];
        v25 = v14[5];
        v26 = v14[4];
        v28 = v14[3];
        sub_1E77870(a1 + 216, 0);
        v20 = *(_DWORD *)(a1 + 224);
        v18 = v24;
        v17 = v25;
        v16 = v26;
        v15 = v28;
      }
      v21 = *(_QWORD *)(a1 + 216) + 56LL * v20;
      if ( v21 )
      {
        *(_QWORD *)v21 = v18;
        *(_QWORD *)(v21 + 8) = v19;
        *(_QWORD *)(v21 + 16) = v15;
        *(_QWORD *)(v21 + 24) = v16;
        *(_QWORD *)(v21 + 40) = v17;
        *(_BYTE *)(v21 + 48) = 1;
        *(_QWORD *)(v21 + 32) = &unk_49FC678;
        v20 = *(_DWORD *)(a1 + 224);
      }
      v22 = *(_QWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 224) = v20 + 1;
      sub_16B7FD0(v22, v18, v19);
      v14 = (_QWORD *)*v14;
    }
    while ( v14 );
  }
  unk_4FC7860 = a1 + 192;
  return qword_4FC7850;
}
