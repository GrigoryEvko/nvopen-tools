// Function: sub_1342A40
// Address: 0x1342a40
//
unsigned __int64 __fastcall sub_1342A40(__int64 a1, _QWORD *a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned int v6; // eax
  char v7; // cl
  unsigned int v8; // eax
  unsigned __int64 v9; // r13
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 result; // rax
  _QWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  unsigned __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  unsigned int v23; // [rsp+1Ch] [rbp-34h]

  v3 = a2[2] & 0xFFFFFFFFFFFFF000LL;
  v4 = sub_130F9A0(v3);
  if ( v4 > 0x7000000000000000LL )
  {
    v9 = 199;
    v13 = 9624;
    v12 = 6400;
    v11 = 9616;
    v10 = 199;
  }
  else
  {
    _BitScanReverse64(&v5, v4);
    v6 = v5 - ((((v4 - 1) & v4) == 0) - 1);
    if ( v6 < 0xE )
      v6 = 14;
    v7 = v6 - 3;
    v8 = v6 - 14;
    if ( !v8 )
      v7 = 12;
    v10 = (((v4 - 1) >> v7) & 3) + 4 * v8;
    v9 = v10;
    v11 = 16LL * (v10 + 402);
    v12 = 32LL * (v10 + 1);
    v13 = v11 + 8;
  }
  v23 = v10;
  --*(_QWORD *)(a1 + v11);
  *(_QWORD *)(a1 + v13) -= v3;
  v22 = a2[4];
  v21 = a2[1];
  sub_1340070((_QWORD *)(a1 + v12), a2);
  if ( sub_133F520((_QWORD *)(a1 + v12)) )
  {
    *(_QWORD *)(a1 + 8 * (v9 >> 6)) &= ~(1LL << v23);
    goto LABEL_9;
  }
  if ( ((*(_QWORD *)(a1 + 32LL * v23 + 56) < v21) - (*(_QWORD *)(a1 + 32LL * v23 + 56) > v21))
     | ((v22 > *(_QWORD *)(a1 + 32LL * v23 + 48)) - (v22 < *(_QWORD *)(a1 + 32LL * v23 + 48))) )
  {
LABEL_9:
    if ( a2 != *(_QWORD **)(a1 + 9632) )
      goto LABEL_10;
    goto LABEL_14;
  }
  v16 = sub_133F530((_QWORD *)(a1 + v12));
  v17 = v16[4];
  v18 = v16[1];
  v19 = (_QWORD *)(a1 + 32 * v9 + 48);
  *v19 = v17;
  v19[1] = v18;
  if ( a2 != *(_QWORD **)(a1 + 9632) )
    goto LABEL_10;
LABEL_14:
  v20 = (_QWORD *)a2[8];
  if ( a2 == v20 )
  {
    *(_QWORD *)(a1 + 9632) = 0;
    goto LABEL_11;
  }
  *(_QWORD *)(a1 + 9632) = v20;
LABEL_10:
  *(_QWORD *)(a2[9] + 64LL) = *(_QWORD *)(a2[8] + 72LL);
  v14 = a2[9];
  *(_QWORD *)(a2[8] + 72LL) = v14;
  a2[9] = *(_QWORD *)(v14 + 64);
  *(_QWORD *)(*(_QWORD *)(a2[8] + 72LL) + 64LL) = a2[8];
  *(_QWORD *)(a2[9] + 64LL) = a2;
LABEL_11:
  result = *(_QWORD *)(a1 + 9640) - (v3 >> 12);
  *(_QWORD *)(a1 + 9640) = result;
  return result;
}
