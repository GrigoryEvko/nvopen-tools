// Function: sub_B01630
// Address: 0xb01630
//
__int64 __fastcall sub_B01630(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r14d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  int v10; // eax
  __int64 v12; // r13
  __int64 v13; // rdi
  unsigned __int8 v14; // al
  __int64 *v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // r14d
  unsigned int v19; // edx
  __int64 *v20; // rsi
  int v21; // r8d
  __int64 *v22; // r9
  int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  __int8 v28[56]; // [rsp+28h] [rbp-38h] BYREF

  v24 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v24;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v25 = 0;
    goto LABEL_7;
  }
  v10 = *(_DWORD *)(a1 + 4);
  v12 = *(_QWORD *)(a3 + 8);
  v13 = a1 - 16;
  LODWORD(v25) = v10;
  HIDWORD(v25) = *(unsigned __int16 *)(v13 + 18);
  v14 = *(_BYTE *)(a1 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(__int64 **)(a1 - 32);
  else
    v15 = (__int64 *)(v13 - 8LL * ((v14 >> 2) & 0xF));
  v26 = *v15;
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v16 = *(_DWORD *)(a1 - 24);
  else
    v16 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v17 = 0;
  if ( v16 == 2 )
    v17 = *((_QWORD *)sub_A17150((_BYTE *)v13) + 1);
  v27 = v17;
  v18 = v4 - 1;
  v28[0] = *(_BYTE *)(a1 + 1) >> 7;
  v8 = v24;
  v19 = v18 & sub_AF71E0((int *)&v25, (int *)&v25 + 1, &v26, &v27, v28);
  v20 = (__int64 *)(v12 + 8LL * v19);
  result = *v20;
  if ( v24 != *v20 )
  {
    v21 = 1;
    v7 = 0;
    while ( result != -4096 )
    {
      if ( result != -8192 || v7 )
        v20 = v7;
      v19 = v18 & (v21 + v19);
      v22 = (__int64 *)(v12 + 8LL * v19);
      result = *v22;
      if ( *v22 == v24 )
        return result;
      ++v21;
      v7 = v20;
      v20 = (__int64 *)(v12 + 8LL * v19);
    }
    v23 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v20;
    ++*(_QWORD *)a3;
    v9 = v23 + 1;
    v25 = v7;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_B01330(a3, v6);
      sub_AFC140(a3, &v24, &v25);
      v7 = v25;
      v8 = v24;
      v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
      *(_DWORD *)(a3 + 16) = v9;
      if ( *v7 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v7 = v8;
      return v24;
    }
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
