// Function: sub_2C0E180
// Address: 0x2c0e180
//
__int64 __fastcall sub_2C0E180(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 *v7; // r13
  __int64 *i; // r14
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = sub_22077B0(0x68u);
  v4 = v1;
  if ( v1 )
  {
    *(_BYTE *)(v1 + 8) = 3;
    v5 = v1 + 64;
    v6 = *(_QWORD *)(a1 + 96);
    *(_QWORD *)(v5 - 40) = 0;
    *(_QWORD *)(v5 - 32) = 0;
    *(_QWORD *)(v5 - 48) = 0;
    *(_QWORD *)(v4 + 48) = v5;
    *(_QWORD *)(v4 + 56) = 0x200000000LL;
    v13[0] = 0;
    *(_QWORD *)(v4 + 80) = 0;
    *(_QWORD *)v4 = &unk_4A23A70;
    *(_QWORD *)(v4 + 40) = &unk_4A23AA8;
    *(_QWORD *)(v4 + 88) = 0;
    sub_9C6650(v13);
    *(_QWORD *)(v4 + 96) = v6;
    *(_QWORD *)v4 = &unk_4A23C10;
    *(_QWORD *)(v4 + 40) = &unk_4A23C60;
  }
  v7 = *(__int64 **)(a1 + 48);
  for ( i = &v7[*(unsigned int *)(a1 + 56)]; i != v7; ++*(_DWORD *)(v10 + 24) )
  {
    v9 = *(unsigned int *)(v4 + 56);
    v10 = *v7;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 60) )
    {
      sub_C8D5F0(v4 + 48, (const void *)(v4 + 64), v9 + 1, 8u, v2, v3);
      v9 = *(unsigned int *)(v4 + 56);
    }
    *(_QWORD *)(*(_QWORD *)(v4 + 48) + 8 * v9) = v10;
    ++*(_DWORD *)(v4 + 56);
    v11 = *(unsigned int *)(v10 + 24);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 28) )
    {
      sub_C8D5F0(v10 + 16, (const void *)(v10 + 32), v11 + 1, 8u, v2, v3);
      v11 = *(unsigned int *)(v10 + 24);
    }
    ++v7;
    *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8 * v11) = v4 + 40;
  }
  return v4;
}
