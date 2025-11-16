// Function: sub_15BEDB0
// Address: 0x15bedb0
//
__int64 __fastcall sub_15BEDB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  unsigned int v11; // r12d
  unsigned int v12; // edx
  __int64 *v13; // r8
  __int64 v14; // rcx
  int v15; // r9d
  __int64 *v16; // r10
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v19; // [rsp+10h] [rbp-30h] BYREF
  __int64 v20[5]; // [rsp+18h] [rbp-28h] BYREF

  v18 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v18;
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
  LODWORD(v19) = *(_DWORD *)(a1 + 28);
  BYTE4(v19) = *(_BYTE *)(a1 + 52);
  v20[0] = *(_QWORD *)(a1 + 8 * (3LL - *(unsigned int *)(a1 + 8)));
  v8 = v18;
  v12 = v11 & sub_15B3730((int *)&v19, (__int8 *)&v19 + 4, v20);
  v13 = (__int64 *)(v10 + 8LL * v12);
  result = v18;
  v14 = *v13;
  if ( *v13 == v18 )
    return result;
  v15 = 1;
  v7 = 0;
  while ( v14 != -8 )
  {
    if ( v14 != -16 || v7 )
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
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_15BEB90(a3, v6);
  sub_15B80C0(a3, &v18, &v19);
  v7 = v19;
  v8 = v18;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v18;
}
