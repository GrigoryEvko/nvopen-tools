// Function: sub_325F8B0
// Address: 0x325f8b0
//
__int64 __fastcall sub_325F8B0(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // r8d
  __int64 v9; // r8
  int v10; // eax
  __int64 v11; // rsi
  int v12; // edx
  unsigned int v13; // eax
  __int64 *v14; // rcx
  __int64 v15; // rdi
  __int64 result; // rax
  int v17; // eax
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  __int64 *v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // rax
  int v27; // r9d
  int v28; // ecx
  int v29; // r9d
  int v30; // ecx
  int v31; // r9d
  __int64 v32[2]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_DWORD *)(a1 + 584);
  v32[0] = a2;
  if ( v3 )
  {
    v17 = *(_DWORD *)(a1 + 592);
    v18 = *(_QWORD *)(a1 + 576);
    v9 = v32[0];
    if ( v17 )
    {
      v19 = v17 - 1;
      v20 = (v17 - 1) & ((LODWORD(v32[0]) >> 9) ^ (LODWORD(v32[0]) >> 4));
      v21 = (__int64 *)(v18 + 8LL * v20);
      v22 = *v21;
      if ( *v21 == v32[0] )
      {
LABEL_15:
        *v21 = -8192;
        v23 = *(unsigned int *)(a1 + 608);
        --*(_DWORD *)(a1 + 584);
        v24 = *(_QWORD **)(a1 + 600);
        ++*(_DWORD *)(a1 + 588);
        v25 = (__int64)&v24[v23];
        v26 = sub_325EB50(v24, v25, v32);
        if ( v26 + 1 != (_QWORD *)v25 )
        {
          memmove(v26, v26 + 1, v25 - (_QWORD)(v26 + 1));
          v27 = *(_DWORD *)(a1 + 608);
          v9 = v32[0];
        }
        *(_DWORD *)(a1 + 608) = v27 - 1;
      }
      else
      {
        v30 = 1;
        while ( v22 != -4096 )
        {
          v31 = v30 + 1;
          v20 = v19 & (v30 + v20);
          v21 = (__int64 *)(v18 + 8LL * v20);
          v22 = *v21;
          if ( *v21 == v32[0] )
            goto LABEL_15;
          v30 = v31;
        }
      }
    }
  }
  else
  {
    v4 = *(unsigned int *)(a1 + 608);
    v5 = *(_QWORD **)(a1 + 600);
    v6 = (__int64)&v5[v4];
    v7 = sub_325EB50(v5, v6, v32);
    if ( (_QWORD *)v6 != v7 )
    {
      if ( (_QWORD *)v6 != v7 + 1 )
      {
        memmove(v7, v7 + 1, v6 - (_QWORD)(v7 + 1));
        v8 = *(_DWORD *)(a1 + 608);
      }
      *(_DWORD *)(a1 + 608) = v8 - 1;
    }
    v9 = v32[0];
  }
  v10 = *(_DWORD *)(a1 + 896);
  v11 = *(_QWORD *)(a1 + 880);
  if ( v10 )
  {
    v12 = v10 - 1;
    v13 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v14 = (__int64 *)(v11 + 24LL * v13);
    v15 = *v14;
    if ( *v14 == v9 )
    {
LABEL_9:
      *v14 = -8192;
      --*(_DWORD *)(a1 + 888);
      ++*(_DWORD *)(a1 + 892);
    }
    else
    {
      v28 = 1;
      while ( v15 != -4096 )
      {
        v29 = v28 + 1;
        v13 = v12 & (v28 + v13);
        v14 = (__int64 *)(v11 + 24LL * v13);
        v15 = *v14;
        if ( v9 == *v14 )
          goto LABEL_9;
        v28 = v29;
      }
    }
  }
  result = *(int *)(v9 + 88);
  if ( (int)result >= 0 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * result) = 0;
    *(_DWORD *)(v9 + 88) = -1;
  }
  return result;
}
