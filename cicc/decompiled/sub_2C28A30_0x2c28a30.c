// Function: sub_2C28A30
// Address: 0x2c28a30
//
__int64 __fastcall sub_2C28A30(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v20; // [rsp+28h] [rbp-48h] BYREF
  __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+38h] [rbp-38h] BYREF

  v20 = *a5;
  if ( v20 )
    sub_2C25AB0(&v20);
  v7 = sub_22077B0(0xA8u);
  if ( v7 )
  {
    v21 = v20;
    if ( v20 )
    {
      sub_2C25AB0(&v21);
      v22[0] = v21;
      if ( v21 )
        sub_2C25AB0(v22);
    }
    else
    {
      v22[0] = 0;
    }
    v8 = *(unsigned int *)(a3 + 24);
    v9 = *(unsigned int *)(a3 + 28);
    *(_BYTE *)(v7 + 8) = 10;
    *(_QWORD *)(v7 + 64) = a3;
    v10 = v7 + 40;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 56) = 0x200000001LL;
    *(_QWORD *)v7 = &unk_4A231A8;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 40) = &unk_4A23170;
    *(_QWORD *)(v7 + 48) = v7 + 64;
    if ( v8 + 1 > v9 )
    {
      sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), v8 + 1, 8u, v10, v6);
      v8 = *(unsigned int *)(a3 + 24);
      v10 = v7 + 40;
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v8) = v10;
    ++*(_DWORD *)(a3 + 24);
    *(_QWORD *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 40) = &unk_4A23AA8;
    v11 = v22[0];
    *(_QWORD *)v7 = &unk_4A23A70;
    *(_QWORD *)(v7 + 88) = v11;
    if ( v11 )
      sub_2C25AB0((__int64 *)(v7 + 88));
    sub_9C6650(v22);
    sub_2BF0340(v7 + 96, 1, 0, v7, v12, v13);
    *(_QWORD *)v7 = &unk_4A231C8;
    *(_QWORD *)(v7 + 40) = &unk_4A23200;
    *(_QWORD *)(v7 + 96) = &unk_4A23238;
    sub_9C6650(&v21);
    *(_DWORD *)(v7 + 152) = a2;
    *(_QWORD *)v7 = &unk_4A24560;
    *(_QWORD *)(v7 + 40) = &unk_4A245A0;
    *(_QWORD *)(v7 + 96) = &unk_4A245D8;
    *(_QWORD *)(v7 + 160) = a4;
  }
  if ( *a1 )
  {
    v14 = (__int64 *)a1[1];
    *(_QWORD *)(v7 + 80) = *a1;
    v15 = *(_QWORD *)(v7 + 24);
    v16 = *v14;
    *(_QWORD *)(v7 + 32) = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v7 + 24) = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v7 + 24;
    *v14 = *v14 & 7 | (v7 + 24);
  }
  sub_9C6650(&v20);
  return v7;
}
