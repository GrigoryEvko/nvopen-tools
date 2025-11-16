// Function: sub_216A710
// Address: 0x216a710
//
__int64 __fastcall sub_216A710(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned __int8 v6; // bl
  __int64 v7; // r8
  unsigned int v8; // r14d
  unsigned int v9; // eax
  char v10; // al
  unsigned __int64 v11; // rax
  unsigned int v12; // r9d
  unsigned int v13; // r8d
  __int64 v14; // rcx
  _QWORD *v15; // r13
  unsigned int v16; // r12d
  int v17; // ebx
  _QWORD *v18; // rax
  int v19; // r14d
  int v20; // ebx
  __int64 v21; // rcx
  int v22; // r13d
  int v23; // ebx
  int v24; // r14d
  int i; // r13d
  __int64 v26; // rdx
  unsigned int v28; // [rsp+Ch] [rbp-64h]
  int v29; // [rsp+10h] [rbp-60h]
  int v30; // [rsp+14h] [rbp-5Ch]
  __int64 v31; // [rsp+18h] [rbp-58h]
  int v32; // [rsp+18h] [rbp-58h]
  __int64 *v33; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  int v35; // [rsp+30h] [rbp-40h]
  unsigned int v36; // [rsp+34h] [rbp-3Ch]
  int v37; // [rsp+38h] [rbp-38h]
  __int64 v38; // [rsp+38h] [rbp-38h]
  int v39; // [rsp+38h] [rbp-38h]

  v4 = a3;
  v5 = a2;
  v6 = a4;
  v7 = *(_QWORD *)(a2 + 32);
  v28 = -1;
  v8 = v7;
  v34 = **(_QWORD **)(a2 + 16);
  v33 = **(__int64 ***)(a3 + 16);
  if ( (_DWORD)v7 )
  {
    _BitScanReverse(&v9, v7);
    v28 = 31 - (v9 ^ 0x1F);
  }
  v10 = *(_BYTE *)(a2 + 8);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(v34 + 8);
  v31 = *(_QWORD *)(a2 + 32);
  v37 = ((unsigned __int8)(v10 - 1) < 6u) + 51;
  v11 = sub_1F43D80(a1[2], *a1, a2, a4);
  v36 = 1;
  v12 = v37;
  v13 = v31;
  if ( (unsigned __int8)(BYTE4(v11) - 14) <= 0x5Fu )
    v36 = word_4328F20[(unsigned __int8)(BYTE4(v11) - 14)];
  v32 = v6 + 1;
  if ( v36 >= v13 )
  {
    v19 = 0;
  }
  else
  {
    v29 = 0;
    v14 = v4;
    v15 = (_QWORD *)a2;
    v16 = v37;
    v30 = 0;
    v35 = 0;
    do
    {
      v30 += v32;
      v38 = v14;
      v8 >>= 1;
      v17 = sub_2169B90(a1, v16, (__int64)v15, v14, 0);
      v35 += sub_2169B90(a1, 0x37u, (__int64)v15, v38, 0) + v17;
      v15 = sub_16463B0((__int64 *)v34, v8);
      v18 = sub_16463B0(v33, v8);
      ++v29;
      v14 = (__int64)v18;
    }
    while ( v36 < v8 );
    v28 -= v29;
    v12 = v16;
    v5 = (__int64)v15;
    v4 = (__int64)v18;
    v19 = v35 + v30;
  }
  v20 = sub_2169B90(a1, v12, v5, v4, 0);
  v22 = v32 + v20 + sub_2169B90(a1, 0x37u, v5, v4, 0);
  v23 = 0;
  v39 = v28 * v22 + v19;
  v24 = *(_QWORD *)(v5 + 32);
  if ( v24 > 0 )
  {
    for ( i = 0; i != v24; ++i )
    {
      v26 = v5;
      if ( *(_BYTE *)(v5 + 8) == 16 )
        v26 = **(_QWORD **)(v5 + 16);
      v23 += sub_1F43D80(a1[2], *a1, v26, v21);
    }
    v23 *= 3;
  }
  return v23 + v39 + (unsigned int)sub_2169B90(a1, 0x37u, v34, (__int64)v33, 0);
}
