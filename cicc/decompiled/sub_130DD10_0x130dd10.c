// Function: sub_130DD10
// Address: 0x130dd10
//
__int64 __fastcall sub_130DD10(__int64 a1)
{
  int v1; // r15d
  int v2; // r13d
  _BOOL4 v3; // r14d
  int v4; // ebx
  __int64 v5; // r12
  int v6; // r8d
  int v7; // r13d
  int v8; // r15d
  _BOOL4 v9; // r15d
  int v10; // r14d
  __int64 v11; // r13
  int v12; // r9d
  __int64 v13; // r10
  int v14; // r12d
  int v15; // eax
  __int64 v16; // r14
  int v17; // r13d
  int v18; // r15d
  __int64 v19; // rbx
  int v20; // esi
  __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v25; // [rsp+10h] [rbp-70h]
  int v26; // [rsp+1Ch] [rbp-64h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  int v28; // [rsp+28h] [rbp-58h]
  int v29; // [rsp+2Ch] [rbp-54h]
  int v30; // [rsp+30h] [rbp-50h]
  int v31; // [rsp+34h] [rbp-4Ch]
  __int64 v32; // [rsp+38h] [rbp-48h]
  int v33; // [rsp+40h] [rbp-40h]
  int v34; // [rsp+44h] [rbp-3Ch]
  __int64 v35; // [rsp+48h] [rbp-38h]

  sub_130DC30(a1 + 76, 0, 3, 3, 0);
  v1 = *(unsigned __int8 *)(a1 + 92);
  v2 = *(unsigned __int8 *)(a1 + 93);
  v3 = *(_DWORD *)(a1 + 100) != 0;
  sub_130DC30(a1 + 104, 1, 3, 3, 1);
  v4 = v1 - ((*(_BYTE *)(a1 + 120) == 0) - 1);
  v5 = a1 + 132;
  v6 = 1;
  v7 = v2 - ((*(_BYTE *)(a1 + 121) == 0) - 1);
  do
  {
    v8 = v6 + 1;
    sub_130DC30(v5, v6 + 1, 4, 4, v6);
    v6 = v8;
    v4 -= (*(_BYTE *)(v5 + 16) == 0) - 1;
    v7 -= (*(_BYTE *)(v5 + 17) == 0) - 1;
    v5 += 28;
  }
  while ( v8 != 4 );
  v31 = v4;
  v28 = 0;
  v9 = v3;
  v27 = 0;
  v26 = 5;
  v29 = v7;
  v10 = 6;
  v11 = 0;
  while ( 1 )
  {
    v12 = v10 - 2;
    v33 = v10;
    v25 = v26;
    v13 = 1;
    v34 = (v10 != 62) + 3;
    v14 = v26;
    v32 = 1LL << v10;
    v15 = v10 + 1;
    v16 = v11;
    v17 = v9;
    v30 = v15;
    v18 = v12;
    v19 = a1 + 28LL * v26 + 76;
    do
    {
      v20 = v14;
      v35 = v13;
      ++v14;
      sub_130DC30(v19, v20, v33, v18, v13);
      v21 = v32 + (v35 << v18);
      if ( *(_DWORD *)(v19 + 24) )
      {
        v16 = v32 + (v35 << v18);
        v17 = v14;
      }
      v31 -= (*(_BYTE *)(v19 + 16) == 0) - 1;
      if ( *(_BYTE *)(v19 + 17) )
      {
        ++v29;
        v27 = v32 + (v35 << v18);
        v28 = v30;
      }
      v19 += 28;
      v13 = v35 + 1;
    }
    while ( v34 >= (int)v35 + 1 );
    v26 += v34;
    v9 = v17;
    v11 = v16;
    if ( v30 == 63 )
      break;
    v10 = v30;
  }
  _BitScanReverse64(&v22, v26);
  *(_DWORD *)(a1 + 12) = v26;
  *(_DWORD *)a1 = 1;
  *(_DWORD *)(a1 + 4) = v9;
  *(_DWORD *)(a1 + 16) = v22 - (((v26 & (v25 + (unsigned int)(v34 - 1))) == 0) - 1);
  *(_DWORD *)(a1 + 8) = v29;
  *(_DWORD *)(a1 + 20) = v31;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 40) = v27;
  *(_QWORD *)(a1 + 32) = v16;
  *(_DWORD *)(a1 + 48) = v28;
  *(_QWORD *)(a1 + 56) = 1LL << v28;
  *(_QWORD *)(a1 + 64) = v21;
  *(_BYTE *)(a1 + 72) = 1;
  return 1LL << v28;
}
