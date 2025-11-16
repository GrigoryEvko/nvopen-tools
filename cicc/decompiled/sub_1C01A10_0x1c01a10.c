// Function: sub_1C01A10
// Address: 0x1c01a10
//
__int64 __fastcall sub_1C01A10(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  unsigned int v4; // r8d
  __int64 v5; // rdi
  unsigned int v6; // r9d
  unsigned int v7; // r10d
  __int64 result; // rax
  unsigned int *v9; // rdx
  unsigned int v10; // ecx
  unsigned int *v11; // r11
  int v12; // r13d
  int v13; // esi
  int v14; // ecx
  int v15; // eax
  unsigned int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // ecx
  int *v19; // rdx
  int v20; // edi
  unsigned int v21; // r13d
  unsigned int v22; // r14d
  int v23; // r11d
  int v24; // r11d
  int *v25; // r10
  int v26; // ecx
  int v27; // edi
  int v28; // r14d
  unsigned int *v29; // r11
  int v30; // eax
  int v31; // r15d
  unsigned int v32; // [rsp+Ch] [rbp-44h] BYREF
  int v33; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v34[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 8;
  v4 = *(_DWORD *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  v32 = a2;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 8);
    v12 = *(_DWORD *)a1;
    goto LABEL_6;
  }
  v6 = v4 - 1;
  v7 = a2;
  result = (v4 - 1) & (37 * a2);
  v9 = (unsigned int *)(v5 + 8 * result);
  v10 = *v9;
  v11 = v9;
  if ( a2 != *v9 )
  {
    v21 = *v9;
    v22 = result;
    v23 = 1;
    while ( v21 != -1 )
    {
      v31 = v23 + 1;
      v22 = v6 & (v23 + v22);
      v11 = (unsigned int *)(v5 + 8LL * v22);
      v21 = *v11;
      if ( a2 == *v11 )
        goto LABEL_3;
      v23 = v31;
    }
    v12 = *(_DWORD *)a1;
    v7 = a2;
    LODWORD(result) = v6 & (37 * a2);
    v9 = (unsigned int *)(v5 + 8LL * (unsigned int)result);
    v10 = *v9;
LABEL_12:
    if ( v10 == a2 )
      goto LABEL_13;
    v28 = 1;
    v29 = 0;
    while ( v10 != -1 )
    {
      if ( v10 == -2 && !v29 )
        v29 = v9;
      LODWORD(result) = v6 & (v28 + result);
      v9 = (unsigned int *)(v5 + 8LL * (unsigned int)result);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_13;
      ++v28;
    }
    v30 = *(_DWORD *)(a1 + 24);
    if ( v29 )
      v9 = v29;
    ++*(_QWORD *)(a1 + 8);
    v14 = v30 + 1;
    if ( 4 * (v30 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 28) - v14 > v4 >> 3 )
        goto LABEL_8;
      v13 = v4;
LABEL_7:
      sub_1BFDD60(v3, v13);
      sub_1BFD720(v3, (int *)&v32, v34);
      v9 = (unsigned int *)v34[0];
      v7 = v32;
      v14 = *(_DWORD *)(a1 + 24) + 1;
LABEL_8:
      *(_DWORD *)(a1 + 24) = v14;
      if ( *v9 != -1 )
        --*(_DWORD *)(a1 + 28);
      *v9 = v7;
      v9[1] = 0;
LABEL_13:
      v9[1] = v12;
      v15 = *(_DWORD *)a1;
      v16 = *(_DWORD *)(a1 + 64);
      v33 = *(_DWORD *)a1;
      if ( v16 )
      {
        v17 = *(_QWORD *)(a1 + 48);
        v18 = (v16 - 1) & (37 * v15);
        v19 = (int *)(v17 + 8LL * v18);
        v20 = *v19;
        if ( v15 == *v19 )
        {
LABEL_15:
          result = v32;
          v19[1] = v32;
          ++*(_DWORD *)a1;
          return result;
        }
        v24 = 1;
        v25 = 0;
        while ( v20 != 0x7FFFFFFF )
        {
          if ( !v25 && v20 == 0x80000000 )
            v25 = v19;
          v18 = (v16 - 1) & (v24 + v18);
          v19 = (int *)(v17 + 8LL * v18);
          v20 = *v19;
          if ( v15 == *v19 )
            goto LABEL_15;
          ++v24;
        }
        v26 = *(_DWORD *)(a1 + 56);
        if ( v25 )
          v19 = v25;
        ++*(_QWORD *)(a1 + 40);
        v27 = v26 + 1;
        if ( 4 * (v26 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 60) - v27 > v16 >> 3 )
          {
LABEL_25:
            *(_DWORD *)(a1 + 56) = v27;
            if ( *v19 != 0x7FFFFFFF )
              --*(_DWORD *)(a1 + 60);
            *v19 = v15;
            v19[1] = 0;
            goto LABEL_15;
          }
LABEL_30:
          sub_1C01850(a1 + 40, v16);
          sub_1BFD870(a1 + 40, &v33, v34);
          v19 = (int *)v34[0];
          v15 = v33;
          v27 = *(_DWORD *)(a1 + 56) + 1;
          goto LABEL_25;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 40);
      }
      v16 *= 2;
      goto LABEL_30;
    }
LABEL_6:
    v13 = 2 * v4;
    goto LABEL_7;
  }
LABEL_3:
  if ( v11 == (unsigned int *)(v5 + 8LL * v4) )
  {
    v12 = *(_DWORD *)a1;
    goto LABEL_12;
  }
  return result;
}
