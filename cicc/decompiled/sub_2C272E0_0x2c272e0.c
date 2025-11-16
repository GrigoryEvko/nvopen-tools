// Function: sub_2C272E0
// Address: 0x2c272e0
//
void *__fastcall sub_2C272E0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 40;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 8) = 16;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = a3;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  v10 = a3 + 16;
  v11 = *(unsigned int *)(a3 + 24);
  v12 = *(unsigned int *)(a3 + 28);
  v17 = 0;
  v18 = 0;
  v19[0] = 0;
  if ( v11 + 1 > v12 )
  {
    sub_C8D5F0(v10, (const void *)(a3 + 32), v11 + 1, 8u, v11 + 1, a6);
    v11 = *(unsigned int *)(a3 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v11) = v6;
  ++*(_DWORD *)(a3 + 24);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v13 = v19[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v13;
  if ( v13 )
    sub_2C25AB0((__int64 *)(a1 + 88));
  sub_9C6650(v19);
  sub_2BF0340(a1 + 96, 1, 0, a1, v14, v15);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v18);
  *(_BYTE *)(a1 + 152) = 7;
  *(_DWORD *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  sub_9C6650(&v17);
  *(_DWORD *)(a1 + 160) = a2;
  *(_QWORD *)(a1 + 168) = a4;
  *(_QWORD *)a1 = &unk_4A23F58;
  *(_QWORD *)(a1 + 40) = &unk_4A23F90;
  *(_QWORD *)(a1 + 96) = &unk_4A23FC8;
  return &unk_4A23FC8;
}
