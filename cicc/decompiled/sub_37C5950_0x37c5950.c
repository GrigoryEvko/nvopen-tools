// Function: sub_37C5950
// Address: 0x37c5950
//
__int64 __fastcall sub_37C5950(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r14d
  char v4; // al
  __int64 v5; // r9
  __int64 v6; // r12
  unsigned int v8; // esi
  int v9; // eax
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1 + 32;
  v15 = a2;
  v3 = *(_DWORD *)(a1 + 8);
  v4 = sub_37BD550(a1 + 32, &v15, &v16);
  v6 = v16;
  if ( v4 )
    return *(unsigned int *)(v6 + 8);
  v8 = *(_DWORD *)(a1 + 56);
  v9 = *(_DWORD *)(a1 + 48);
  v17[0] = v16;
  ++*(_QWORD *)(a1 + 32);
  v10 = v9 + 1;
  v11 = 2 * v8;
  if ( 4 * v10 >= 3 * v8 )
  {
    v8 *= 2;
    goto LABEL_11;
  }
  if ( v8 - *(_DWORD *)(a1 + 52) - v10 <= v8 >> 3 )
  {
LABEL_11:
    sub_37C5570(v2, v8);
    sub_37BD550(v2, &v15, v17);
    v6 = v17[0];
    v10 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v10;
  if ( unk_5051170 != *(_QWORD *)v6 )
    --*(_DWORD *)(a1 + 52);
  v12 = v15;
  *(_DWORD *)(v6 + 8) = 2 * v3;
  *(_QWORD *)v6 = v12;
  v13 = *(unsigned int *)(a1 + 8);
  v14 = v15;
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 8u, v11, v5);
    v13 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v13) = v14;
  ++*(_DWORD *)(a1 + 8);
  return *(unsigned int *)(v6 + 8);
}
