// Function: sub_216A510
// Address: 0x216a510
//
__int64 __fastcall sub_216A510(__int64 *a1, unsigned int a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // rdi
  unsigned int v9; // ebx
  unsigned int v10; // eax
  unsigned __int64 v11; // rax
  _QWORD *v12; // r10
  unsigned int v13; // r11d
  unsigned __int8 v14; // al
  int v15; // r15d
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // r10
  int v21; // r13d
  __int64 v22; // rax
  unsigned int v23; // r13d
  int v24; // r12d
  __int64 *v25; // r13
  int v26; // r15d
  int v27; // r14d
  __int64 v28; // rbx
  __int64 v29; // rdx
  unsigned __int64 v31; // rax
  __int64 *v32; // [rsp+8h] [rbp-58h]
  unsigned int v33; // [rsp+10h] [rbp-50h]
  unsigned int v34; // [rsp+14h] [rbp-4Ch]
  int v36; // [rsp+18h] [rbp-48h]
  unsigned int v37; // [rsp+18h] [rbp-48h]
  unsigned int v38; // [rsp+20h] [rbp-40h]
  int v39; // [rsp+20h] [rbp-40h]
  int v40; // [rsp+24h] [rbp-3Ch]
  unsigned int v41; // [rsp+28h] [rbp-38h]
  _QWORD *v42; // [rsp+28h] [rbp-38h]
  unsigned int v43; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a3 + 32);
  v7 = *a1;
  v8 = a1[2];
  v9 = v6;
  v32 = **(__int64 ***)(a3 + 16);
  v40 = a4 + 1;
  if ( (_DWORD)v6 )
  {
    _BitScanReverse(&v10, v6);
    v38 = a2;
    v33 = 31 - (v10 ^ 0x1F);
    v11 = sub_1F43D80(v8, v7, a3, a4);
    v41 = 1;
    v12 = (_QWORD *)a3;
    v13 = v38;
    v14 = BYTE4(v11) - 14;
    if ( (unsigned __int8)(BYTE4(v11) - 14) > 0x5Fu )
      goto LABEL_3;
  }
  else
  {
    v37 = a2;
    v31 = sub_1F43D80(v8, v7, a3, a4);
    v33 = -1;
    v12 = (_QWORD *)a3;
    v13 = v37;
    v14 = BYTE4(v31) - 14;
    if ( (unsigned __int8)(BYTE4(v31) - 14) > 0x5Fu )
      goto LABEL_7;
  }
  v41 = word_4328F20[v14];
LABEL_3:
  if ( v41 >= (unsigned int)v6 )
  {
    v9 = 0;
  }
  else
  {
    v36 = 0;
    v15 = 0;
    v16 = v12;
    v39 = 0;
    v34 = v13;
    do
    {
      v9 >>= 1;
      ++v15;
      v36 += v40;
      v39 += sub_216EC30(a1, v34, v16, 0, 0, 0, 0, 0, 0);
      v17 = sub_16463B0(v32, v9);
      v16 = v17;
    }
    while ( v41 < v9 );
    v33 -= v15;
    v13 = v34;
    v12 = v17;
    v9 = v39 + v36;
  }
LABEL_7:
  v42 = v12;
  v18 = sub_216EC30(a1, v13, v12, 0, 0, 0, 0, 0, 0);
  v20 = (__int64)v42;
  v21 = v33 * (v18 + v40);
  v22 = v42[4];
  v23 = v9 + v21;
  if ( (int)v22 > 0 )
  {
    v43 = v23;
    v24 = 0;
    v25 = a1;
    v26 = 0;
    v27 = v22;
    v28 = v20;
    do
    {
      v29 = v28;
      if ( *(_BYTE *)(v28 + 8) == 16 )
        v29 = **(_QWORD **)(v28 + 16);
      ++v26;
      v24 += sub_1F43D80(v25[2], *v25, v29, v19);
    }
    while ( v27 != v26 );
    return v24 + v43;
  }
  return v23;
}
