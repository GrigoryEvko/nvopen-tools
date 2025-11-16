// Function: sub_15BC690
// Address: 0x15bc690
//
__int64 __fastcall sub_15BC690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r14
  unsigned int v11; // r12d
  unsigned int v12; // edx
  __int64 *v13; // rsi
  int v14; // r8d
  __int64 *v15; // r9
  int v16; // eax
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19; // [rsp+18h] [rbp-38h] BYREF
  __int64 v20; // [rsp+20h] [rbp-30h] BYREF
  int v21[10]; // [rsp+28h] [rbp-28h] BYREF

  v17 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v17;
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
  v10 = *(_QWORD *)(a3 + 8);
  v11 = v4 - 1;
  LODWORD(v18) = *(unsigned __int16 *)(a1 + 2);
  v19 = *(_QWORD *)(a1 + 8 * (2LL - *(unsigned int *)(a1 + 8)));
  v20 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)v21 = *(_QWORD *)(a1 + 48);
  v8 = v17;
  v12 = v11 & sub_15B4F20((int *)&v18, &v19, &v20, v21, &v21[1]);
  v13 = (__int64 *)(v10 + 8LL * v12);
  result = *v13;
  if ( v17 == *v13 )
    return result;
  v14 = 1;
  v7 = 0;
  while ( result != -8 )
  {
    if ( result != -16 || v7 )
      v13 = v7;
    v12 = v11 & (v14 + v12);
    v15 = (__int64 *)(v10 + 8LL * v12);
    result = *v15;
    if ( v17 == *v15 )
      return result;
    ++v14;
    v7 = v13;
    v13 = (__int64 *)(v10 + 8LL * v12);
  }
  v16 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v13;
  ++*(_QWORD *)a3;
  v9 = v16 + 1;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_15BC460(a3, v6);
  sub_15B78B0(a3, &v17, &v18);
  v7 = v18;
  v8 = v17;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v17;
}
