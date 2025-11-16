// Function: sub_15C42B0
// Address: 0x15c42b0
//
__int64 __fastcall sub_15C42B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  unsigned int v10; // r12d
  __int64 v11; // r13
  int v12; // eax
  unsigned int v13; // edx
  __int64 *v14; // r8
  __int64 v15; // rcx
  int v16; // r9d
  __int64 *v17; // r10
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-28h] BYREF

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
  v10 = v4 - 1;
  v11 = *(_QWORD *)(a3 + 8);
  v12 = sub_15B1DB0(*(__int64 **)(a1 + 24), *(_QWORD *)(a1 + 32));
  v8 = v19;
  v13 = v10 & v12;
  v14 = (__int64 *)(v11 + 8LL * (v10 & v12));
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
    v13 = v10 & (v16 + v13);
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
  sub_15C40B0(a3, v6);
  sub_15B9210(a3, &v19, &v20);
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
