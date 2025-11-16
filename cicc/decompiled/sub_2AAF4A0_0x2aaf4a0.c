// Function: sub_2AAF4A0
// Address: 0x2aaf4a0
//
void *__fastcall sub_2AAF4A0(__int64 a1, char a2, __int64 *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *v8; // r15
  __int64 v9; // r8
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // [rsp+0h] [rbp-50h]
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v19[0] = *a5;
  if ( v19[0] )
    sub_2AAAFA0(v19);
  *(_BYTE *)(a1 + 8) = a2;
  v8 = &a3[a4];
  *(_QWORD *)(a1 + 24) = 0;
  v9 = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( v8 != a3 )
  {
    v10 = *a3;
    v11 = a1 + 64;
    v12 = a3 + 1;
    v13 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v11 + 8 * v13) = v10;
      ++*(_DWORD *)(a1 + 56);
      v14 = *(unsigned int *)(v10 + 24);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
      {
        v18 = v9;
        sub_C8D5F0(v10 + 16, (const void *)(v10 + 32), v14 + 1, 8u, v9, a6);
        v14 = *(unsigned int *)(v10 + 24);
        v9 = v18;
      }
      *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v14) = v9;
      ++*(_DWORD *)(v10 + 24);
      if ( v8 == v12 )
        break;
      v13 = *(unsigned int *)(a1 + 56);
      v10 = *v12;
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v17 = v9;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v13 + 1, 8u, v9, a6);
        v13 = *(unsigned int *)(a1 + 56);
        v9 = v17;
      }
      v11 = *(_QWORD *)(a1 + 48);
      ++v12;
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v15 = v19[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v15;
  if ( v15 )
  {
    sub_2AAAFA0((__int64 *)(a1 + 88));
    if ( v19[0] )
      sub_B91220((__int64)v19, v19[0]);
  }
  sub_2BF0340(a1 + 96, 1, 0, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  return &unk_4A23238;
}
