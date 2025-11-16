// Function: sub_2C0E300
// Address: 0x2c0e300
//
__int64 __fastcall sub_2C0E300(__int64 a1)
{
  __int64 *v1; // r15
  __int64 *v2; // r13
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r12
  __int64 v6; // r10
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  int v18; // [rsp+2Ch] [rbp-44h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  __int64 v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 48);
  v2 = &v1[*(unsigned int *)(a1 + 56)];
  v19 = *(_QWORD *)(a1 + 88);
  if ( v19 )
    sub_2AAAFA0(&v19);
  v5 = sub_22077B0(0x68u);
  if ( v5 )
  {
    v18 = *(_DWORD *)(a1 + 96);
    v20[0] = v19;
    if ( v19 )
      sub_2AAAFA0(v20);
    *(_QWORD *)(v5 + 24) = 0;
    v6 = v5 + 40;
    *(_QWORD *)(v5 + 32) = 0;
    *(_BYTE *)(v5 + 8) = 26;
    *(_QWORD *)v5 = &unk_4A231A8;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 48) = v5 + 64;
    *(_QWORD *)(v5 + 40) = &unk_4A23170;
    *(_QWORD *)(v5 + 56) = 0x200000000LL;
    if ( v1 != v2 )
    {
      v7 = *v1;
      v8 = v5 + 64;
      v9 = v1 + 1;
      v10 = 0;
      while ( 1 )
      {
        *(_QWORD *)(v8 + 8 * v10) = v7;
        ++*(_DWORD *)(v5 + 56);
        v11 = *(unsigned int *)(v7 + 24);
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 28) )
        {
          v15 = v6;
          v17 = v7;
          sub_C8D5F0(v7 + 16, (const void *)(v7 + 32), v11 + 1, 8u, v3, v4);
          v7 = v17;
          v6 = v15;
          v11 = *(unsigned int *)(v17 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v7 + 16) + 8 * v11) = v6;
        ++*(_DWORD *)(v7 + 24);
        if ( v2 == v9 )
          break;
        v10 = *(unsigned int *)(v5 + 56);
        v7 = *v9;
        v3 = v10 + 1;
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 60) )
        {
          v14 = *v9;
          v16 = v6;
          sub_C8D5F0(v5 + 48, (const void *)(v5 + 64), v10 + 1, 8u, v3, v4);
          v10 = *(unsigned int *)(v5 + 56);
          v7 = v14;
          v6 = v16;
        }
        v8 = *(_QWORD *)(v5 + 48);
        ++v9;
      }
    }
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 40) = &unk_4A23AA8;
    v12 = v20[0];
    *(_QWORD *)v5 = &unk_4A23A70;
    *(_QWORD *)(v5 + 88) = v12;
    if ( v12 )
      sub_2AAAFA0((__int64 *)(v5 + 88));
    sub_9C6650(v20);
    *(_QWORD *)(v5 + 40) = &unk_4A23DF8;
    *(_QWORD *)v5 = &unk_4A23DC0;
    *(_DWORD *)(v5 + 96) = v18;
  }
  sub_9C6650(&v19);
  return v5;
}
