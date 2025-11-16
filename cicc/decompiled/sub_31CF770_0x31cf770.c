// Function: sub_31CF770
// Address: 0x31cf770
//
__int64 __fastcall sub_31CF770(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  int v7; // eax
  int v8; // eax
  int v10; // eax
  int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // r9
  int v16; // r10d
  unsigned int v17; // ecx
  __int64 v18; // rsi
  __int64 v19; // rdx
  int v20; // ecx
  __int64 v21; // rdi
  int v22; // r10d
  unsigned int v23; // eax
  __int64 v24; // rsi

  v5 = *(_DWORD *)(a1 + 24);
  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v8 = v7 + 1;
  if ( 4 * v8 >= 3 * v5 )
  {
    sub_2CF9C20(a1, 2 * v5);
    v10 = *(_DWORD *)(a1 + 24);
    a3 = 0;
    if ( v10 )
    {
      v19 = *(_QWORD *)(a2 + 24);
      v20 = v10 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      v22 = 1;
      v23 = (v10 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      a3 = v21 + 48LL * v23;
      v24 = *(_QWORD *)(a3 + 24);
      if ( v19 != v24 )
      {
        while ( v24 != -4096 )
        {
          if ( v24 == -8192 && !v15 )
            v15 = a3;
          v23 = v20 & (v22 + v23);
          a3 = v21 + 48LL * v23;
          v24 = *(_QWORD *)(a3 + 24);
          if ( v19 == v24 )
            goto LABEL_7;
          ++v22;
        }
        goto LABEL_11;
      }
    }
  }
  else
  {
    if ( v5 - *(_DWORD *)(a1 + 20) - v8 > v5 >> 3 )
      goto LABEL_3;
    sub_2CF9C20(a1, v5);
    v11 = *(_DWORD *)(a1 + 24);
    a3 = 0;
    if ( v11 )
    {
      v12 = *(_QWORD *)(a2 + 24);
      v13 = v11 - 1;
      v14 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      v16 = 1;
      v17 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      a3 = v14 + 48LL * v17;
      v18 = *(_QWORD *)(a3 + 24);
      if ( v18 != v12 )
      {
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v15 )
            v15 = a3;
          v17 = v13 & (v16 + v17);
          a3 = v14 + 48LL * v17;
          v18 = *(_QWORD *)(a3 + 24);
          if ( v12 == v18 )
            goto LABEL_7;
          ++v16;
        }
LABEL_11:
        if ( v15 )
          a3 = v15;
      }
    }
  }
LABEL_7:
  v8 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *(_QWORD *)(a3 + 24) != -4096 )
    --*(_DWORD *)(a1 + 20);
  return a3;
}
