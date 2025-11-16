// Function: sub_2C15940
// Address: 0x2c15940
//
__int64 __fastcall sub_2C15940(__int64 a1)
{
  __int64 v1; // r13
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rax
  __int16 v14; // [rsp+6h] [rbp-5Ah]
  __int64 v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h] BYREF
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = 0;
  v2 = *(__int64 **)(a1 + 48);
  v3 = *(_QWORD *)(a1 + 96);
  v4 = *v2;
  if ( *(_BYTE *)(a1 + 106) )
    v1 = v2[*(_DWORD *)(a1 + 56) - 1];
  v15 = *(_QWORD *)(a1 + 88);
  if ( v15 )
    sub_2AAAFA0(&v15);
  v6 = sub_22077B0(0xA8u);
  if ( v6 )
  {
    v14 = *(_WORD *)(a1 + 104);
    v17 = v15;
    if ( v15 )
    {
      sub_2AAAFA0(&v17);
      v16 = v4;
      v18[0] = v17;
      if ( v17 )
        sub_2AAAFA0(v18);
    }
    else
    {
      v16 = v4;
      v18[0] = 0;
    }
    sub_2AAF310(v6, 20, &v16, 1, v18, v5);
    sub_9C6650(v18);
    *(_QWORD *)(v6 + 96) = v3;
    *(_BYTE *)(v6 + 106) = 0;
    *(_QWORD *)(v6 + 40) = &unk_4A24740;
    *(_QWORD *)v6 = &unk_4A24708;
    *(_WORD *)(v6 + 104) = v14;
    sub_9C6650(&v17);
    sub_2BF0340(v6 + 112, 1, v3, v6, v7, v8);
    *(_QWORD *)v6 = &unk_4A24778;
    *(_QWORD *)(v6 + 40) = &unk_4A247B8;
    *(_QWORD *)(v6 + 112) = &unk_4A247F0;
    if ( v1 )
    {
      v11 = *(unsigned int *)(v6 + 56);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
      {
        sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v11 + 1, 8u, v9, v10);
        v11 = *(unsigned int *)(v6 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v11) = v1;
      ++*(_DWORD *)(v6 + 56);
      v12 = *(unsigned int *)(v1 + 24);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
      {
        sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v12 + 1, 8u, v9, v10);
        v12 = *(unsigned int *)(v1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v12) = v6 + 40;
      ++*(_DWORD *)(v1 + 24);
      *(_BYTE *)(v6 + 106) = 1;
    }
  }
  sub_9C6650(&v15);
  return v6;
}
