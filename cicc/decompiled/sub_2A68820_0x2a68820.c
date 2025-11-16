// Function: sub_2A68820
// Address: 0x2a68820
//
__int64 __fastcall sub_2A68820(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v4; // r14
  unsigned int v7; // r8d
  __int64 v8; // r9
  __int64 v9; // rdi
  int v10; // ebx
  __int64 *v11; // r10
  unsigned int v12; // edx
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  int v17; // eax
  int v18; // edx
  __int64 v19; // rsi
  int v20; // esi
  unsigned __int8 *v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-28h] BYREF

  v4 = a1 + 136;
  v7 = *(_DWORD *)(a1 + 160);
  v22 = a2;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 136);
    v23 = 0;
LABEL_19:
    v21 = a3;
    v20 = 2 * v7;
LABEL_20:
    sub_2A68410(v4, v20);
    sub_2A65730(v4, &v22, &v23);
    v19 = v22;
    v11 = v23;
    a3 = v21;
    v18 = *(_DWORD *)(a1 + 152) + 1;
    goto LABEL_15;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)(a1 + 144);
  v10 = 1;
  v11 = 0;
  v12 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 48LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
  {
LABEL_3:
    v15 = (__int64)(v13 + 1);
    return sub_2A63320(a1, v15, a2, a3, 0, v8);
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = v8 & (v10 + v12);
    v13 = (__int64 *)(v9 + 48LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
      goto LABEL_3;
    ++v10;
  }
  v17 = *(_DWORD *)(a1 + 152);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)(a1 + 136);
  v18 = v17 + 1;
  v23 = v11;
  if ( 4 * (v17 + 1) >= 3 * v7 )
    goto LABEL_19;
  v19 = a2;
  if ( v7 - *(_DWORD *)(a1 + 156) - v18 <= v7 >> 3 )
  {
    v21 = a3;
    v20 = v7;
    goto LABEL_20;
  }
LABEL_15:
  *(_DWORD *)(a1 + 152) = v18;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 156);
  *v11 = v19;
  v15 = (__int64)(v11 + 1);
  *((_WORD *)v11 + 4) = 0;
  return sub_2A63320(a1, v15, a2, a3, 0, v8);
}
