// Function: sub_2C11ED0
// Address: 0x2c11ed0
//
__int64 __fastcall sub_2C11ED0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  char *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // [rsp+10h] [rbp-60h]
  int v17; // [rsp+1Ch] [rbp-54h]
  __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-40h] BYREF
  char v21; // [rsp+40h] [rbp-30h] BYREF

  v1 = *(__int64 **)(a1 + 48);
  v2 = *v1;
  v3 = v1[1];
  v16 = *(_QWORD *)(a1 + 136);
  v6 = sub_22077B0(0xA0u);
  if ( v6 )
  {
    v7 = *(_DWORD *)(a1 + 152);
    v20[1] = v3;
    v8 = 0;
    v20[0] = v2;
    v9 = v6 + 64;
    v10 = (char *)v20;
    v17 = v7;
    v18 = 0;
    *(_BYTE *)(v6 + 8) = 8;
    *(_QWORD *)v6 = &unk_4A231A8;
    v19 = 0;
    *(_QWORD *)(v6 + 48) = v6 + 64;
    *(_QWORD *)(v6 + 40) = &unk_4A23170;
    *(_QWORD *)(v6 + 56) = 0x200000000LL;
    *(_QWORD *)(v6 + 24) = 0;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 16) = 0;
    while ( 1 )
    {
      *(_QWORD *)(v9 + 8 * v8) = v2;
      ++*(_DWORD *)(v6 + 56);
      v11 = *(unsigned int *)(v2 + 24);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 28) )
      {
        sub_C8D5F0(v2 + 16, (const void *)(v2 + 32), v11 + 1, 8u, v4, v5);
        v11 = *(unsigned int *)(v2 + 24);
      }
      v10 += 8;
      *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8 * v11) = v6 + 40;
      ++*(_DWORD *)(v2 + 24);
      if ( v10 == &v21 )
        break;
      v8 = *(unsigned int *)(v6 + 56);
      v2 = *(_QWORD *)v10;
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
      {
        sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v8 + 1, 8u, v4, v5);
        v8 = *(unsigned int *)(v6 + 56);
      }
      v9 = *(_QWORD *)(v6 + 48);
    }
    *(_QWORD *)(v6 + 80) = 0;
    *(_QWORD *)(v6 + 40) = &unk_4A23AA8;
    v12 = v19;
    *(_QWORD *)v6 = &unk_4A23A70;
    *(_QWORD *)(v6 + 88) = v12;
    if ( v12 )
      sub_2AAAFA0((__int64 *)(v6 + 88));
    sub_9C6650(&v19);
    sub_2BF0340(v6 + 96, 1, v16, v6, v13, v14);
    *(_QWORD *)v6 = &unk_4A231C8;
    *(_QWORD *)(v6 + 40) = &unk_4A23200;
    *(_QWORD *)(v6 + 96) = &unk_4A23238;
    sub_9C6650(&v18);
    *(_QWORD *)v6 = &unk_4A23AE0;
    *(_QWORD *)(v6 + 96) = &unk_4A23B50;
    *(_QWORD *)(v6 + 40) = &unk_4A23B18;
    *(_DWORD *)(v6 + 152) = v17;
    sub_2BF0490(*(_QWORD *)(*(_QWORD *)(v6 + 48) + 8LL));
  }
  return v6;
}
