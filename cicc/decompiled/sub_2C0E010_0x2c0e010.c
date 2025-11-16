// Function: sub_2C0E010
// Address: 0x2c0e010
//
__int64 __fastcall sub_2C0E010(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r9
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = **(_QWORD **)(a1 + 48);
  v7 = *(_QWORD *)(a1 + 88);
  if ( v7 )
    sub_2AAAFA0(&v7);
  v3 = sub_22077B0(0x60u);
  if ( v3 )
  {
    v8[0] = v7;
    if ( v7 )
      sub_2AAAFA0(v8);
    *(_BYTE *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 24) = 0;
    *(_QWORD *)(v3 + 64) = v1;
    *(_QWORD *)v3 = &unk_4A231A8;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 40) = &unk_4A23170;
    *(_QWORD *)(v3 + 48) = v3 + 64;
    *(_QWORD *)(v3 + 56) = 0x200000001LL;
    v4 = *(unsigned int *)(v1 + 24);
    if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
    {
      sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v4 + 1, 8u, v4 + 1, v2);
      v4 = *(unsigned int *)(v1 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v4) = v3 + 40;
    ++*(_DWORD *)(v1 + 24);
    *(_QWORD *)(v3 + 80) = 0;
    *(_QWORD *)(v3 + 40) = &unk_4A23AA8;
    v5 = v8[0];
    *(_QWORD *)v3 = &unk_4A23A70;
    *(_QWORD *)(v3 + 88) = v5;
    if ( v5 )
      sub_2AAAFA0((__int64 *)(v3 + 88));
    sub_9C6650(v8);
    *(_QWORD *)v3 = &unk_4A245F8;
    *(_QWORD *)(v3 + 40) = &unk_4A24638;
  }
  sub_9C6650(&v7);
  return v3;
}
