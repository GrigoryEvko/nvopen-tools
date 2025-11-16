// Function: sub_2C15E50
// Address: 0x2c15e50
//
__int64 __fastcall sub_2C15E50(__int64 a1)
{
  __int64 v1; // rbx
  __int64 *v2; // rax
  __int64 v3; // r14
  __int64 v4; // r9
  __int64 v5; // r15
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  __int16 v11; // [rsp+Eh] [rbp-72h]
  __int64 v12; // [rsp+10h] [rbp-70h]
  __int64 v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+28h] [rbp-58h] BYREF
  __int64 v15; // [rsp+30h] [rbp-50h] BYREF
  __int64 v16; // [rsp+38h] [rbp-48h] BYREF
  __int64 v17; // [rsp+40h] [rbp-40h] BYREF
  __int64 v18; // [rsp+48h] [rbp-38h]

  v1 = 0;
  v13 = *(_QWORD *)(a1 + 96);
  v2 = *(__int64 **)(a1 + 48);
  v3 = *v2;
  v12 = v2[1];
  if ( *(_BYTE *)(a1 + 106) )
    v1 = v2[*(_DWORD *)(a1 + 56) - 1];
  v14 = *(_QWORD *)(a1 + 88);
  if ( v14 )
    sub_2AAAFA0(&v14);
  v5 = sub_22077B0(0x70u);
  if ( v5 )
  {
    v11 = *(_WORD *)(a1 + 104);
    v15 = v14;
    if ( v14 )
    {
      sub_2AAAFA0(&v15);
      v17 = v3;
      v18 = v12;
      v16 = v15;
      if ( v15 )
        sub_2AAAFA0(&v16);
    }
    else
    {
      v17 = v3;
      v16 = 0;
      v18 = v12;
    }
    sub_2AAF310(v5, 22, &v17, 2, &v16, v4);
    sub_9C6650(&v16);
    *(_BYTE *)(v5 + 106) = 0;
    *(_QWORD *)(v5 + 40) = &unk_4A24740;
    *(_QWORD *)v5 = &unk_4A24708;
    *(_QWORD *)(v5 + 96) = v13;
    *(_WORD *)(v5 + 104) = v11;
    sub_9C6650(&v15);
    *(_QWORD *)v5 = &unk_4A248A8;
    *(_QWORD *)(v5 + 40) = &unk_4A248E8;
    if ( v1 )
    {
      v8 = *(unsigned int *)(v5 + 56);
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 60) )
      {
        sub_C8D5F0(v5 + 48, (const void *)(v5 + 64), v8 + 1, 8u, v6, v7);
        v8 = *(unsigned int *)(v5 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v5 + 48) + 8 * v8) = v1;
      ++*(_DWORD *)(v5 + 56);
      v9 = *(unsigned int *)(v1 + 24);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
      {
        sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v9 + 1, 8u, v6, v7);
        v9 = *(unsigned int *)(v1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v9) = v5 + 40;
      ++*(_DWORD *)(v1 + 24);
      *(_BYTE *)(v5 + 106) = 1;
    }
  }
  sub_9C6650(&v14);
  return v5;
}
