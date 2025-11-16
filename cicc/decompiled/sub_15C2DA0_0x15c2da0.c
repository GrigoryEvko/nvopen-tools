// Function: sub_15C2DA0
// Address: 0x15c2da0
//
__int64 __fastcall sub_15C2DA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned int v12; // r12d
  unsigned int v13; // edx
  __int64 *v14; // r8
  __int64 v15; // rcx
  int v16; // r9d
  __int64 *v17; // r10
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-78h] BYREF
  __int64 *v20; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+18h] [rbp-68h] BYREF
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h] BYREF
  int v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h] BYREF
  __int8 v26; // [rsp+40h] [rbp-40h] BYREF
  __int8 v27[7]; // [rsp+41h] [rbp-3Fh] BYREF
  __int64 v28; // [rsp+48h] [rbp-38h] BYREF
  int v29; // [rsp+50h] [rbp-30h]

  v19 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v19;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)(a3 + 8);
  v12 = v4 - 1;
  v20 = *(__int64 **)(a1 - 8 * v10);
  v21 = *(_QWORD *)(a1 + 8 * (1 - v10));
  v22 = *(_QWORD *)(a1 + 8 * (5 - v10));
  v23 = *(_QWORD *)(a1 + 8 * (2 - v10));
  v24 = *(_DWORD *)(a1 + 24);
  v25 = *(_QWORD *)(a1 + 8 * (3 - v10));
  v26 = *(_BYTE *)(a1 + 32);
  v27[0] = *(_BYTE *)(a1 + 33);
  v28 = *(_QWORD *)(a1 + 8 * (6 - v10));
  v29 = *(_DWORD *)(a1 + 28);
  v8 = v19;
  v13 = v12 & sub_15B44C0((__int64 *)&v20, &v21, &v22, &v23, &v24, &v25, &v26, v27, &v28);
  v14 = (__int64 *)(v11 + 8LL * v13);
  result = v19;
  v15 = *v14;
  if ( *v14 == v19 )
    return result;
  v16 = 1;
  v7 = 0;
  while ( v15 != -8 )
  {
    if ( v15 != -16 || v7 )
      v14 = v7;
    v13 = v12 & (v16 + v13);
    v17 = (__int64 *)(v11 + 8LL * v13);
    v15 = *v17;
    if ( *v17 == v19 )
      return result;
    ++v16;
    v7 = v14;
    v14 = (__int64 *)(v11 + 8LL * v13);
  }
  v18 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v14;
  ++*(_QWORD *)a3;
  v9 = v18 + 1;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_15C2C50(a3, v6);
  sub_15B8E40(a3, &v19, &v20);
  v7 = v20;
  v8 = v19;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v19;
}
