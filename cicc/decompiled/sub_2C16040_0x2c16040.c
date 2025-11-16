// Function: sub_2C16040
// Address: 0x2c16040
//
__int64 __fastcall sub_2C16040(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // r9
  __int64 v8; // r10
  int v9; // r15d
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v17; // [rsp+4h] [rbp-5Ch]
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h] BYREF
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 136);
  v2 = **(_QWORD **)(a1 + 48);
  v3 = sub_22077B0(0xA8u);
  v6 = v3;
  if ( v3 )
  {
    v20 = v2;
    v7 = *(unsigned __int16 *)(a1 + 160);
    v8 = *(_QWORD *)(a1 + 152);
    v9 = *(_DWORD *)(a1 + 164);
    v19 = 0;
    v17 = v7;
    v18 = v8;
    v21 = 0;
    v22[0] = 0;
    sub_2AAF310(v3, 36, &v20, 1, v22, v7);
    sub_9C6650(v22);
    sub_2BF0340(v6 + 96, 1, v1, v6, v10, v11);
    *(_QWORD *)v6 = &unk_4A231C8;
    *(_QWORD *)(v6 + 40) = &unk_4A23200;
    *(_QWORD *)(v6 + 96) = &unk_4A23238;
    sub_9C6650(&v21);
    *(_QWORD *)v6 = &unk_4A23FE8;
    *(_QWORD *)(v6 + 40) = &unk_4A24030;
    *(_QWORD *)(v6 + 96) = &unk_4A24068;
    sub_9C6650(&v19);
    *(_DWORD *)(v6 + 164) = v9;
    v5 = v17;
    *(_QWORD *)(v6 + 152) = v18;
    *(_QWORD *)v6 = &unk_4A24C80;
    *(_QWORD *)(v6 + 40) = &unk_4A24CD0;
    *(_QWORD *)(v6 + 96) = &unk_4A24D08;
    *(_WORD *)(v6 + 160) = v17;
  }
  v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL);
  if ( v12 == sub_2AA7530 )
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  else
    v13 = v12(a1);
  v14 = *(unsigned int *)(v6 + 56);
  if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
  {
    sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v14 + 1, 8u, v4, v5);
    v14 = *(unsigned int *)(v6 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v14) = v13;
  ++*(_DWORD *)(v6 + 56);
  v15 = *(unsigned int *)(v13 + 24);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 28) )
  {
    sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v15 + 1, 8u, v4, v5);
    v15 = *(unsigned int *)(v13 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v15) = v6 + 40;
  ++*(_DWORD *)(v13 + 24);
  return v6;
}
