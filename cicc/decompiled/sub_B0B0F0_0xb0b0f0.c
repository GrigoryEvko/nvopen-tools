// Function: sub_B0B0F0
// Address: 0xb0b0f0
//
__int64 __fastcall sub_B0B0F0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r14d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned int v17; // r14d
  unsigned int v18; // edx
  __int64 *v19; // rsi
  int v20; // r8d
  __int64 *v21; // r9
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+18h] [rbp-58h] BYREF
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int8 v27[8]; // [rsp+28h] [rbp-48h] BYREF
  __int64 v28[8]; // [rsp+30h] [rbp-40h] BYREF

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
  v10 = *(_QWORD *)(a3 + 8);
  LODWORD(v24) = (unsigned __int16)sub_AF18C0(a1);
  v11 = sub_AF5140(a1, 0);
  v12 = a1 - 16;
  v25 = v11;
  v13 = *(_BYTE *)(a1 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(a1 - 32);
  else
    v14 = v12 - 8LL * ((v13 >> 2) & 0xF);
  v26 = *(_QWORD *)(v14 + 8);
  v27[0] = *(_BYTE *)(a1 + 1) >> 7;
  v15 = *(_BYTE *)(a1 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a1 - 32);
  else
    v16 = v12 - 8LL * ((v15 >> 2) & 0xF);
  v17 = v4 - 1;
  v28[0] = *(_QWORD *)(v16 + 16);
  v8 = v23;
  v18 = v17 & sub_AF9230((int *)&v24, &v25, &v26, v27, v28);
  v19 = (__int64 *)(v10 + 8LL * v18);
  result = *v19;
  if ( v23 == *v19 )
    return result;
  v20 = 1;
  v7 = 0;
  while ( result != -4096 )
  {
    if ( result != -8192 || v7 )
      v19 = v7;
    v18 = v17 & (v20 + v18);
    v21 = (__int64 *)(v10 + 8LL * v18);
    result = *v21;
    if ( *v21 == v23 )
      return result;
    ++v20;
    v7 = v19;
    v19 = (__int64 *)(v10 + 8LL * v18);
  }
  v22 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v19;
  ++*(_QWORD *)a3;
  v9 = v22 + 1;
  v24 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B0ADF0(a3, v6);
  sub_AFEAC0(a3, &v23, &v24);
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
