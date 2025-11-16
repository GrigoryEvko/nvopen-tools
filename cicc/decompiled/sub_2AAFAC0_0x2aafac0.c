// Function: sub_2AAFAC0
// Address: 0x2aafac0
//
void *__fastcall sub_2AAFAC0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v12 = *a5;
  if ( v12 )
  {
    sub_2AAAFA0(&v12);
    v13[0] = v12;
    if ( v12 )
      sub_2AAAFA0(v13);
  }
  else
  {
    v13[0] = 0;
  }
  *(_BYTE *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = a4;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  v8 = *(unsigned int *)(a4 + 24);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 28) )
  {
    sub_C8D5F0(a4 + 16, (const void *)(a4 + 32), v8 + 1, 8u, v8 + 1, a6);
    v8 = *(unsigned int *)(a4 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a4 + 16) + 8 * v8) = a1 + 40;
  ++*(_DWORD *)(a4 + 24);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v9 = v13[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v9;
  if ( v9 )
  {
    sub_2AAAFA0((__int64 *)(a1 + 88));
    if ( v13[0] )
      sub_B91220((__int64)v13, v13[0]);
  }
  sub_2BF0340(a1 + 96, 1, a3, a1);
  v10 = v12;
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  if ( v10 )
    sub_B91220((__int64)&v12, v10);
  *(_QWORD *)a1 = &unk_4A23FE8;
  *(_QWORD *)(a1 + 40) = &unk_4A24030;
  *(_QWORD *)(a1 + 96) = &unk_4A24068;
  return &unk_4A24068;
}
