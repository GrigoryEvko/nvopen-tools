// Function: sub_B08E90
// Address: 0xb08e90
//
__int64 __fastcall sub_B08E90(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int v12; // r12d
  unsigned int v13; // edx
  __int64 *v14; // rsi
  int v15; // r8d
  __int64 *v16; // r9
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h] BYREF
  int v21[16]; // [rsp+20h] [rbp-40h] BYREF

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
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v19 = (__int64 *)*((_QWORD *)sub_A17150((_BYTE *)(a1 - 16)) + 1);
  v11 = a1;
  if ( *(_BYTE *)a1 != 16 )
    v11 = *(_QWORD *)sub_A17150((_BYTE *)(a1 - 16));
  v20 = v11;
  v12 = v4 - 1;
  v21[0] = *(_DWORD *)(a1 + 4);
  v8 = v18;
  v13 = v12 & sub_AF7750((__int64 *)&v19, &v20, v21);
  v14 = (__int64 *)(v10 + 8LL * v13);
  result = *v14;
  if ( v18 != *v14 )
  {
    v15 = 1;
    v7 = 0;
    while ( result != -4096 )
    {
      if ( result != -8192 || v7 )
        v14 = v7;
      v13 = v12 & (v15 + v13);
      v16 = (__int64 *)(v10 + 8LL * v13);
      result = *v16;
      if ( *v16 == v18 )
        return result;
      ++v15;
      v7 = v14;
      v14 = (__int64 *)(v10 + 8LL * v13);
    }
    v17 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v14;
    ++*(_QWORD *)a3;
    v9 = v17 + 1;
    v19 = v7;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_B08BF0(a3, v6);
      sub_AFE2D0(a3, &v18, &v19);
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
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
