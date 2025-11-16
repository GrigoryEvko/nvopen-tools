// Function: sub_1342830
// Address: 0x1342830
//
__int64 __fastcall sub_1342830(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r14
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  char v9; // cl
  unsigned int v10; // eax
  int v11; // ecx
  __int64 v12; // r12
  __int64 v13; // rdi
  unsigned __int64 v14; // r13
  bool v15; // al
  _QWORD *v16; // rdi
  unsigned __int64 *v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rcx
  int v21; // eax
  unsigned __int64 *v22; // rax
  unsigned __int64 v23; // [rsp+8h] [rbp-58h]
  unsigned int v24; // [rsp+14h] [rbp-4Ch]
  __int64 v25; // [rsp+20h] [rbp-40h]
  unsigned __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = a2;
  v4 = a2[2] & 0xFFFFFFFFFFFFF000LL;
  v5 = sub_130F9A0(v4);
  if ( v5 > 0x7000000000000000LL )
  {
    v12 = 9616;
    v13 = 6400;
    v11 = 199;
    v25 = 9624;
    v23 = 199;
  }
  else
  {
    v6 = v5 - 1;
    _BitScanReverse64(&v7, v5);
    v8 = v7 - ((((v5 - 1) & v5) == 0) - 1);
    if ( v8 < 0xE )
      v8 = 14;
    v9 = v8 - 3;
    v10 = v8 - 14;
    if ( !v10 )
      v9 = 12;
    v23 = ((v6 >> v9) & 3) + 4 * v10;
    v11 = v23;
    v12 = 16LL * (unsigned int)(v23 + 402);
    v13 = 32LL * (unsigned int)(v23 + 1);
    v25 = v12 + 8;
  }
  v14 = a2[4];
  v24 = v11;
  v26 = a2[1];
  v15 = sub_133F520((_QWORD *)(a1 + v13));
  v16 = (_QWORD *)(a1 + v13);
  if ( v15 )
  {
    *(_QWORD *)(a1 + 8 * (v23 >> 6)) |= 1LL << v24;
    v17 = (unsigned __int64 *)(a1 + 32 * v23 + 48);
    *v17 = v14;
    v17[1] = v26;
  }
  else
  {
    v20 = a1 + 32LL * v24;
    v21 = (*(_QWORD *)(v20 + 48) < v14) - (*(_QWORD *)(v20 + 48) > v14);
    if ( !v21 )
      v21 = (v26 > *(_QWORD *)(v20 + 56)) - (v26 < *(_QWORD *)(v20 + 56));
    if ( v21 == -1 )
    {
      v22 = (unsigned __int64 *)(a1 + 32 * v23 + 48);
      *v22 = v14;
      v22[1] = v26;
    }
  }
  sub_133F890(v16, a2);
  ++*(_QWORD *)(a1 + v12);
  *(_QWORD *)(a1 + v25) += v4;
  a2[8] = a2;
  a2[9] = a2;
  v18 = *(_QWORD *)(a1 + 9632);
  if ( v18 )
  {
    a2[8] = *(_QWORD *)(v18 + 72);
    *(_QWORD *)(*(_QWORD *)(a1 + 9632) + 72LL) = a2;
    a2[9] = *(_QWORD *)(a2[9] + 64LL);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 9632) + 72LL) + 64LL) = *(_QWORD *)(a1 + 9632);
    *(_QWORD *)(a2[9] + 64LL) = a2;
    v3 = (_QWORD *)a2[8];
  }
  *(_QWORD *)(a1 + 9632) = v3;
  result = *(_QWORD *)(a1 + 9640);
  *(_QWORD *)(a1 + 9640) = result + (v4 >> 12);
  return result;
}
