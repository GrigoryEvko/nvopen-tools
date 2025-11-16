// Function: sub_A45AE0
// Address: 0xa45ae0
//
__int64 __fastcall sub_A45AE0(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // r8d
  int v8; // r10d
  __int64 v9; // rcx
  __int64 v10; // r13
  unsigned int v11; // r14d
  unsigned int v12; // edx
  __int64 v13; // rsi
  __int64 v14; // rax
  int v15; // eax
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  _BYTE *v19; // rsi
  _BYTE *v20; // rsi
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  int v31; // r8d
  unsigned int v32; // r14d
  __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35[7]; // [rsp+8h] [rbp-38h] BYREF

  v35[0] = a3;
  if ( !a3 )
    return 0;
  v5 = a1 + 256;
  v7 = *(_DWORD *)(a1 + 280);
  if ( v7 )
  {
    v8 = 1;
    v9 = *(_QWORD *)(a1 + 264);
    v10 = 0;
    v11 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
    v12 = (v7 - 1) & v11;
    v13 = v9 + 16LL * v12;
    v14 = *(_QWORD *)v13;
    if ( a3 == *(_QWORD *)v13 )
    {
LABEL_4:
      v15 = *(_DWORD *)(v13 + 8);
      if ( a2 != v15 )
      {
        if ( v15 )
          sub_A3F4E0(a1, v13);
      }
      return 0;
    }
    while ( v14 != -4096 )
    {
      if ( !v10 && v14 == -8192 )
        v10 = v13;
      v12 = (v7 - 1) & (v8 + v12);
      v13 = v9 + 16LL * v12;
      v14 = *(_QWORD *)v13;
      if ( a3 == *(_QWORD *)v13 )
        goto LABEL_4;
      ++v8;
    }
    v17 = *(_DWORD *)(a1 + 272);
    if ( !v10 )
      v10 = v13;
    ++*(_QWORD *)(a1 + 256);
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 276) - v18 > v7 >> 3 )
        goto LABEL_19;
      sub_A42F50(v5, v7);
      v28 = *(_DWORD *)(a1 + 280);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 264);
        v31 = 1;
        v32 = v29 & v11;
        v18 = *(_DWORD *)(a1 + 272) + 1;
        v33 = 0;
        v10 = v30 + 16LL * v32;
        v34 = *(_QWORD *)v10;
        if ( a3 != *(_QWORD *)v10 )
        {
          while ( v34 != -4096 )
          {
            if ( !v33 && v34 == -8192 )
              v33 = v10;
            v32 = v29 & (v31 + v32);
            v10 = v30 + 16LL * v32;
            v34 = *(_QWORD *)v10;
            if ( a3 == *(_QWORD *)v10 )
              goto LABEL_19;
            ++v31;
          }
          if ( v33 )
            v10 = v33;
        }
        goto LABEL_19;
      }
LABEL_53:
      ++*(_DWORD *)(a1 + 272);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 256);
  }
  sub_A42F50(v5, 2 * v7);
  v21 = *(_DWORD *)(a1 + 280);
  if ( !v21 )
    goto LABEL_53;
  v22 = v21 - 1;
  v23 = *(_QWORD *)(a1 + 264);
  v24 = (v21 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = *(_DWORD *)(a1 + 272) + 1;
  v10 = v23 + 16LL * v24;
  v25 = *(_QWORD *)v10;
  if ( a3 != *(_QWORD *)v10 )
  {
    v26 = 1;
    v27 = 0;
    while ( v25 != -4096 )
    {
      if ( !v27 && v25 == -8192 )
        v27 = v10;
      v24 = v22 & (v26 + v24);
      v10 = v23 + 16LL * v24;
      v25 = *(_QWORD *)v10;
      if ( a3 == *(_QWORD *)v10 )
        goto LABEL_19;
      ++v26;
    }
    if ( v27 )
      v10 = v27;
  }
LABEL_19:
  *(_DWORD *)(a1 + 272) = v18;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 276);
  *(_QWORD *)v10 = a3;
  result = v35[0];
  *(_DWORD *)(v10 + 8) = a2;
  *(_DWORD *)(v10 + 12) = 0;
  if ( (unsigned __int8)(*(_BYTE *)result - 5) > 0x1Fu )
  {
    v19 = *(_BYTE **)(a1 + 216);
    if ( v19 == *(_BYTE **)(a1 + 224) )
    {
      sub_A40280(a1 + 208, v19, v35);
      result = v35[0];
      v20 = *(_BYTE **)(a1 + 216);
    }
    else
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = result;
        v19 = *(_BYTE **)(a1 + 216);
        result = v35[0];
      }
      v20 = v19 + 8;
      *(_QWORD *)(a1 + 216) = v20;
    }
    *(_DWORD *)(v10 + 12) = (__int64)&v20[-*(_QWORD *)(a1 + 208)] >> 3;
    if ( *(_BYTE *)result == 1 )
    {
      sub_A45280(a1, *(unsigned __int8 **)(result + 136));
      return 0;
    }
    return 0;
  }
  return result;
}
