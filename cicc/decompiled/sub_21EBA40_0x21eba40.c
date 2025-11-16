// Function: sub_21EBA40
// Address: 0x21eba40
//
__int64 __fastcall sub_21EBA40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r8
  __int64 result; // rax
  __int64 v12; // r12
  unsigned int v13; // ebx
  _WORD *v14; // rsi
  int v15; // edx
  int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // esi
  unsigned int v20; // r9d
  __int64 v21; // r8
  int v22; // r10d
  unsigned int v23; // ecx
  unsigned int v24; // r11d
  _DWORD *v25; // rdx
  int v26; // edi
  int v27; // edi
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // rax
  int v31; // eax
  int v32; // r11d
  _DWORD *v33; // r10
  int v34; // edi
  int v35; // edi
  int v36; // r9d
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  int v39; // [rsp+24h] [rbp-3Ch] BYREF
  _DWORD *v40; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a1 + 136);
  v7 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v31 = 1;
    while ( v10 != -8 )
    {
      v36 = v31 + 1;
      v8 = (v6 - 1) & (v31 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v31 = v36;
    }
  }
  v9 = (__int64 *)(v7 + 16 * v6);
LABEL_3:
  v37 = v9[1];
  result = sub_1DD5D10(a3);
  v12 = *(_QWORD *)(a3 + 32);
  v38 = result;
  if ( v12 != result )
  {
    while ( 1 )
    {
      v13 = sub_1E163A0(v12);
      result = *(_QWORD *)(v12 + 16);
      v14 = *(_WORD **)(result + 32);
      if ( v14 && *v14 )
      {
        v15 = 0;
        do
          result = (unsigned int)++v15;
        while ( v14[v15] );
        v13 += v15;
      }
      v16 = *(_DWORD *)(v12 + 40);
      if ( v13 != v16 )
        break;
LABEL_14:
      if ( (*(_BYTE *)v12 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
          v12 = *(_QWORD *)(v12 + 8);
      }
      v12 = *(_QWORD *)(v12 + 8);
      if ( v38 == v12 )
        return result;
    }
    while ( 1 )
    {
      v17 = v13;
      v18 = *(_QWORD *)(v12 + 32);
      ++v13;
      result = v18 + 40 * v17;
      if ( *(_BYTE *)result )
        goto LABEL_13;
      result = *(unsigned int *)(result + 8);
      if ( (int)result >= 0 )
        goto LABEL_13;
      if ( a2 != *(_QWORD *)(v18 + 40LL * v13 + 24) )
        goto LABEL_13;
      v19 = *(_DWORD *)(a1 + 80);
      if ( !v19 )
        goto LABEL_13;
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 64);
      v22 = 1;
      v23 = (v19 - 1) & (37 * result);
      v24 = v23;
      v25 = (_DWORD *)(v21 + 8LL * v23);
      v26 = *v25;
      if ( (_DWORD)result != *v25 )
      {
        while ( v26 != -1 )
        {
          v24 = v20 & (v22 + v24);
          v26 = *(_DWORD *)(v21 + 8LL * v24);
          if ( (_DWORD)result == v26 )
            goto LABEL_22;
          ++v22;
        }
        goto LABEL_13;
      }
LABEL_22:
      v39 = result;
      v27 = *v25;
      if ( (_DWORD)result != *v25 )
        break;
LABEL_23:
      v28 = v25[1];
      v29 = 1LL << v28;
      v30 = 8LL * (v28 >> 6);
LABEL_24:
      result = *(_QWORD *)(v37 + 48) + v30;
      *(_QWORD *)result |= v29;
LABEL_13:
      if ( v16 == v13 )
        goto LABEL_14;
    }
    v32 = 1;
    v33 = 0;
    while ( v27 != -1 )
    {
      if ( !v33 && v27 == -2 )
        v33 = v25;
      v23 = v20 & (v32 + v23);
      v25 = (_DWORD *)(v21 + 8LL * v23);
      v27 = *v25;
      if ( (_DWORD)result == *v25 )
        goto LABEL_23;
      ++v32;
    }
    v34 = *(_DWORD *)(a1 + 72);
    if ( v33 )
      v25 = v33;
    ++*(_QWORD *)(a1 + 56);
    v35 = v34 + 1;
    if ( 4 * v35 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(a1 + 76) - v35 > v19 >> 3 )
    {
LABEL_38:
      *(_DWORD *)(a1 + 72) = v35;
      if ( *v25 != -1 )
        --*(_DWORD *)(a1 + 76);
      *v25 = result;
      v30 = 0;
      v25[1] = 0;
      v29 = 1;
      goto LABEL_24;
    }
    sub_1BFDD60(a1 + 56, v19);
    sub_1BFD720(a1 + 56, &v39, &v40);
    v25 = v40;
    LODWORD(result) = v39;
    v35 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_38;
  }
  return result;
}
