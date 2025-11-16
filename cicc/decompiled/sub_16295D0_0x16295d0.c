// Function: sub_16295D0
// Address: 0x16295d0
//
__int64 __fastcall sub_16295D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  int v5; // r9d
  __int64 v6; // r13
  unsigned int v7; // r9d
  int v8; // esi
  __int64 *v9; // rsi
  __int64 v10; // rdi
  int v11; // eax
  __int64 result; // rax
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // rsi
  int i; // r11d
  __int64 v17; // r13
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 *v22; // r8
  __int64 v23; // rcx
  int v24; // r10d
  __int64 *v25; // r11
  int v26; // eax
  __int64 v27; // [rsp+8h] [rbp-78h]
  unsigned int v28; // [rsp+14h] [rbp-6Ch]
  int v29; // [rsp+20h] [rbp-60h]
  int v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v32; // [rsp+30h] [rbp-50h] BYREF
  __int64 v33; // [rsp+38h] [rbp-48h] BYREF
  __int64 v34[8]; // [rsp+40h] [rbp-40h] BYREF

  v31 = a1;
  LODWORD(v32) = *(_DWORD *)(a1 + 4);
  HIDWORD(v32) = *(unsigned __int16 *)(a1 + 2);
  v3 = *(unsigned int *)(a1 + 8);
  v33 = *(_QWORD *)(a1 - 8 * v3);
  v4 = 0;
  if ( (_DWORD)v3 == 2 )
    v4 = *(_QWORD *)(a1 - 8);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v34[0] = v4;
  v29 = v5;
  if ( !v5 )
    goto LABEL_4;
  v13 = (v5 - 1) & sub_15B3100((int *)&v32, (int *)&v32 + 1, &v33, v34);
  v14 = (__int64 *)(v6 + 8LL * v13);
  v15 = *v14;
  if ( *v14 == -8 )
  {
LABEL_17:
    v17 = *(_QWORD *)(a2 + 8);
    LODWORD(v18) = *(_DWORD *)(a2 + 24);
  }
  else
  {
    for ( i = 1; ; ++i )
    {
      if ( v15 != -16 && v32 == (__int64 *)__PAIR64__(*(unsigned __int16 *)(v15 + 2), *(_DWORD *)(v15 + 4)) )
      {
        v28 = *(_DWORD *)(v15 + 8);
        if ( v33 == *(_QWORD *)(v15 - 8LL * v28) )
        {
          v27 = 0;
          if ( v28 == 2 )
            v27 = *(_QWORD *)(v15 - 8);
          if ( v34[0] == v27 )
            break;
        }
      }
      v13 = (v29 - 1) & (i + v13);
      v14 = (__int64 *)(v6 + 8LL * v13);
      v15 = *v14;
      if ( *v14 == -8 )
        goto LABEL_17;
    }
    v17 = *(_QWORD *)(a2 + 8);
    v18 = *(unsigned int *)(a2 + 24);
    if ( v14 != (__int64 *)(v17 + 8 * v18) )
    {
      result = *v14;
      if ( *v14 )
        return result;
    }
  }
  if ( !(_DWORD)v18 )
  {
LABEL_4:
    ++*(_QWORD *)a2;
    v7 = 0;
    goto LABEL_5;
  }
  LODWORD(v32) = *(_DWORD *)(v31 + 4);
  HIDWORD(v32) = *(unsigned __int16 *)(v31 + 2);
  v19 = *(unsigned int *)(v31 + 8);
  v33 = *(_QWORD *)(v31 - 8 * v19);
  v20 = 0;
  if ( (_DWORD)v19 == 2 )
    v20 = *(_QWORD *)(v31 - 8);
  v34[0] = v20;
  v30 = v18;
  v10 = v31;
  LODWORD(v21) = (v18 - 1) & sub_15B3100((int *)&v32, (int *)&v32 + 1, &v33, v34);
  v22 = (__int64 *)(v17 + 8LL * (unsigned int)v21);
  result = v31;
  v23 = *v22;
  if ( *v22 != v31 )
  {
    v24 = 1;
    v9 = 0;
    while ( v23 != -8 )
    {
      if ( v23 != -16 || v9 )
        v22 = v9;
      v21 = (v30 - 1) & (unsigned int)(v21 + v24);
      v25 = (__int64 *)(v17 + 8 * v21);
      v23 = *v25;
      if ( *v25 == v31 )
        return result;
      ++v24;
      v9 = v22;
      v22 = (__int64 *)(v17 + 8 * v21);
    }
    v26 = *(_DWORD *)(a2 + 16);
    v7 = *(_DWORD *)(a2 + 24);
    if ( !v9 )
      v9 = v22;
    ++*(_QWORD *)a2;
    v11 = v26 + 1;
    if ( 4 * v11 < 3 * v7 )
    {
      if ( v7 - (v11 + *(_DWORD *)(a2 + 20)) > v7 >> 3 )
        goto LABEL_7;
      v8 = v7;
LABEL_6:
      sub_15B9A30(a2, v8);
      sub_15B7120(a2, &v31, &v32);
      v9 = v32;
      v10 = v31;
      v11 = *(_DWORD *)(a2 + 16) + 1;
LABEL_7:
      *(_DWORD *)(a2 + 16) = v11;
      if ( *v9 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v9 = v10;
      return v31;
    }
LABEL_5:
    v8 = 2 * v7;
    goto LABEL_6;
  }
  return result;
}
