// Function: sub_B04CA0
// Address: 0xb04ca0
//
__int64 __fastcall sub_B04CA0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r13d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r14
  unsigned int v11; // r13d
  unsigned int v12; // edx
  __int64 *v13; // rdi
  __int64 v14; // rcx
  int v15; // r8d
  __int64 *v16; // r9
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  int v22; // [rsp+28h] [rbp-48h] BYREF
  int v23[17]; // [rsp+2Ch] [rbp-44h] BYREF

  v18 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v18;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v19 = 0;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v11 = v4 - 1;
  LODWORD(v19) = (unsigned __int16)sub_AF18C0(a1);
  v20 = sub_AF5140(a1, 2u);
  v21 = *(_QWORD *)(a1 + 24);
  v22 = sub_AF18D0(a1);
  v23[0] = *(_DWORD *)(a1 + 44);
  v23[1] = *(_DWORD *)(a1 + 40);
  v23[2] = *(_DWORD *)(a1 + 20);
  v8 = v18;
  v12 = v11 & sub_AF9B00((int *)&v19, &v20, &v21, &v22, v23);
  v13 = (__int64 *)(v10 + 8LL * v12);
  result = v18;
  v14 = *v13;
  if ( *v13 == v18 )
    return result;
  v15 = 1;
  v7 = 0;
  while ( v14 != -4096 )
  {
    if ( v14 != -8192 || v7 )
      v13 = v7;
    v12 = v11 & (v15 + v12);
    v16 = (__int64 *)(v10 + 8LL * v12);
    v14 = *v16;
    if ( *v16 == v18 )
      return result;
    ++v15;
    v7 = v13;
    v13 = (__int64 *)(v10 + 8LL * v12);
  }
  v17 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v13;
  ++*(_QWORD *)a3;
  v9 = v17 + 1;
  v19 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B049F0(a3, v6);
  sub_AFCEE0(a3, &v18, &v19);
  v7 = v19;
  v8 = v18;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v18;
}
