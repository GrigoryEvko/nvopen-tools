// Function: sub_15BE5F0
// Address: 0x15be5f0
//
__int64 __fastcall sub_15BE5F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r12d
  unsigned int v14; // edx
  __int64 *v15; // rsi
  int v16; // r8d
  __int64 *v17; // r9
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-78h] BYREF
  __int64 *v20; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+18h] [rbp-68h] BYREF
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  int v23; // [rsp+28h] [rbp-58h] BYREF
  __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25[3]; // [rsp+38h] [rbp-48h] BYREF
  int v26; // [rsp+50h] [rbp-30h]
  int v27; // [rsp+54h] [rbp-2Ch]
  __int64 v28[5]; // [rsp+58h] [rbp-28h] BYREF

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
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a3 + 8);
  LODWORD(v20) = *(unsigned __int16 *)(a1 + 2);
  v11 = *(unsigned int *)(a1 + 8);
  v21 = *(_QWORD *)(a1 + 8 * (2 - v11));
  v12 = a1;
  if ( *(_BYTE *)a1 != 15 )
    v12 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v22 = v12;
  v13 = v4 - 1;
  v23 = *(_DWORD *)(a1 + 24);
  v24 = *(_QWORD *)(a1 + 8 * (1 - v11));
  v25[0] = *(_QWORD *)(a1 + 8 * (3 - v11));
  v25[1] = *(_QWORD *)(a1 + 32);
  v25[2] = *(_QWORD *)(a1 + 40);
  v26 = *(_DWORD *)(a1 + 48);
  v27 = *(_DWORD *)(a1 + 28);
  v28[0] = *(_QWORD *)(a1 + 8 * (4 - v11));
  v8 = v19;
  v14 = v13 & sub_15B5D10(&v21, &v22, &v23, v25, &v24, v28);
  v15 = (__int64 *)(v10 + 8LL * v14);
  result = *v15;
  if ( v19 != *v15 )
  {
    v16 = 1;
    v7 = 0;
    while ( result != -8 )
    {
      if ( result != -16 || v7 )
        v15 = v7;
      v14 = v13 & (v16 + v14);
      v17 = (__int64 *)(v10 + 8LL * v14);
      result = *v17;
      if ( *v17 == v19 )
        return result;
      ++v16;
      v7 = v15;
      v15 = (__int64 *)(v10 + 8LL * v14);
    }
    v18 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v15;
    ++*(_QWORD *)a3;
    v9 = v18 + 1;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_15BE350(a3, v6);
      sub_15B7F60(a3, &v19, &v20);
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
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
