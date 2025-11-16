// Function: sub_B0C7D0
// Address: 0xb0c7d0
//
__int64 __fastcall sub_B0C7D0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r14d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  unsigned __int8 v10; // al
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 *v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // r15
  unsigned int v16; // r14d
  unsigned int v17; // edx
  __int64 *v18; // r8
  __int64 v19; // rcx
  int v20; // r9d
  __int64 *v21; // r10
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25[2]; // [rsp+18h] [rbp-48h] BYREF
  int v26[14]; // [rsp+28h] [rbp-38h] BYREF

  v23 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v23;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v24 = 0;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(_BYTE *)(a1 - 16);
  v11 = *(_QWORD *)(a3 + 8);
  v12 = a1 - 16;
  if ( (v10 & 2) != 0 )
    v13 = *(__int64 **)(a1 - 32);
  else
    v13 = (__int64 *)(v12 - 8LL * ((v10 >> 2) & 0xF));
  v24 = (__int64 *)*v13;
  v25[0] = sub_AF5140(a1, 1u);
  v14 = *(_BYTE *)(a1 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(a1 - 32);
  else
    v15 = v12 - 8LL * ((v14 >> 2) & 0xF);
  v16 = v4 - 1;
  v25[1] = *(_QWORD *)(v15 + 16);
  v26[0] = *(_DWORD *)(a1 + 4);
  v8 = v23;
  v17 = v16 & sub_AF8830((__int64 *)&v24, v25, v26);
  v18 = (__int64 *)(v11 + 8LL * v17);
  result = v23;
  v19 = *v18;
  if ( *v18 == v23 )
    return result;
  v20 = 1;
  v7 = 0;
  while ( v19 != -4096 )
  {
    if ( v19 != -8192 || v7 )
      v18 = v7;
    v17 = v16 & (v20 + v17);
    v21 = (__int64 *)(v11 + 8LL * v17);
    v19 = *v21;
    if ( *v21 == v23 )
      return result;
    ++v20;
    v7 = v18;
    v18 = (__int64 *)(v11 + 8LL * v17);
  }
  v22 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v18;
  ++*(_QWORD *)a3;
  v9 = v22 + 1;
  v24 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B0C500(a3, v6);
  sub_AFF130(a3, &v23, &v24);
  v7 = v24;
  v8 = v23;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v23;
}
