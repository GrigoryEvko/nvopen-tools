// Function: sub_1623100
// Address: 0x1623100
//
__int64 __fastcall sub_1623100(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r10
  int v5; // r9d
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  int v16; // r12d
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  __m128i v18; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v19[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
  {
    v4 = a1 + 32;
    v5 = 3;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 40);
    v4 = *(_QWORD *)(a1 + 32);
    if ( !(_DWORD)v13 )
      goto LABEL_8;
    v5 = v13 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 24LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v15 = 1;
    while ( v8 != -4 )
    {
      v16 = v15 + 1;
      v6 = v5 & (v15 + v6);
      v7 = (__int64 *)(v4 + 24LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_4;
      v15 = v16;
    }
    if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
    {
      v14 = 96;
      goto LABEL_9;
    }
    v13 = *(unsigned int *)(a1 + 40);
LABEL_8:
    v14 = 24 * v13;
LABEL_9:
    v7 = (__int64 *)(v4 + v14);
  }
LABEL_4:
  v9 = v7[1];
  v10 = v7[2];
  *v7 = -8;
  v11 = *(_DWORD *)(a1 + 24);
  ++*(_DWORD *)(a1 + 28);
  v17 = a3;
  v18.m128i_i64[0] = v9;
  v18.m128i_i64[1] = v10;
  *(_DWORD *)(a1 + 24) = (2 * (v11 >> 1) - 2) | v11 & 1;
  return sub_1622EA0((__int64)v19, a1 + 16, &v17, &v18);
}
