// Function: sub_162C000
// Address: 0x162c000
//
__int64 __fastcall sub_162C000(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v4; // rdx
  int v5; // ecx
  __int64 v6; // r13
  unsigned int v7; // esi
  __int64 *v8; // r8
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  int v12; // esi
  int v13; // eax
  unsigned int v14; // esi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rdi
  int v22; // r13d
  __int64 v23; // rdx
  __int64 *v24; // r9
  __int64 v25; // rsi
  int v26; // r10d
  __int64 *v27; // r11
  int v28; // eax
  int i; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v34; // [rsp+20h] [rbp-50h] BYREF
  __int64 v35; // [rsp+28h] [rbp-48h] BYREF
  int v36[16]; // [rsp+30h] [rbp-40h] BYREF

  v2 = a1;
  v4 = *(unsigned int *)(a1 + 8);
  v33 = a1;
  v34 = *(__int64 **)(a1 + 8 * (1 - v4));
  if ( *(_BYTE *)a1 != 15 )
    v2 = *(_QWORD *)(a1 - 8 * v4);
  v35 = v2;
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v30 = v5;
  v36[0] = *(_DWORD *)(a1 + 24);
  if ( !v5 )
    goto LABEL_4;
  v12 = sub_15B2A30((__int64 *)&v34, &v35, v36);
  v13 = v30 - 1;
  v14 = (v30 - 1) & v12;
  v15 = (__int64 *)(v6 + 8LL * v14);
  v16 = *v15;
  if ( *v15 == -8 )
  {
LABEL_17:
    v18 = *(_QWORD *)(a2 + 8);
    LODWORD(v19) = *(_DWORD *)(a2 + 24);
  }
  else
  {
    for ( i = 1; ; ++i )
    {
      if ( v16 != -16 )
      {
        v17 = *(unsigned int *)(v16 + 8);
        v31 = v16;
        if ( v34 == *(__int64 **)(v16 + 8 * (1 - v17)) )
        {
          if ( *(_BYTE *)v16 != 15 )
            v31 = *(_QWORD *)(v16 - 8 * v17);
          if ( v35 == v31 && v36[0] == *(_DWORD *)(v16 + 24) )
            break;
        }
      }
      v14 = v13 & (i + v14);
      v15 = (__int64 *)(v6 + 8LL * v14);
      v16 = *v15;
      if ( *v15 == -8 )
        goto LABEL_17;
    }
    v18 = *(_QWORD *)(a2 + 8);
    v19 = *(unsigned int *)(a2 + 24);
    if ( v15 != (__int64 *)(v18 + 8 * v19) )
    {
      result = *v15;
      if ( *v15 )
        return result;
    }
  }
  if ( !(_DWORD)v19 )
  {
LABEL_4:
    ++*(_QWORD *)a2;
    v7 = 0;
    goto LABEL_5;
  }
  v20 = *(unsigned int *)(v33 + 8);
  v21 = v33;
  v34 = *(__int64 **)(v33 + 8 * (1 - v20));
  if ( *(_BYTE *)v33 != 15 )
    v21 = *(_QWORD *)(v33 - 8 * v20);
  v35 = v21;
  v32 = v18;
  v22 = v19 - 1;
  v36[0] = *(_DWORD *)(v33 + 24);
  v9 = v33;
  LODWORD(v23) = v22 & sub_15B2A30((__int64 *)&v34, &v35, v36);
  v24 = (__int64 *)(v32 + 8LL * (unsigned int)v23);
  result = v33;
  v25 = *v24;
  if ( *v24 != v33 )
  {
    v26 = 1;
    v8 = 0;
    while ( v25 != -8 )
    {
      if ( v25 != -16 || v8 )
        v24 = v8;
      v23 = v22 & (unsigned int)(v23 + v26);
      v27 = (__int64 *)(v32 + 8 * v23);
      v25 = *v27;
      if ( *v27 == v33 )
        return result;
      ++v26;
      v8 = v24;
      v24 = (__int64 *)(v32 + 8 * v23);
    }
    v28 = *(_DWORD *)(a2 + 16);
    v7 = *(_DWORD *)(a2 + 24);
    if ( !v8 )
      v8 = v24;
    ++*(_QWORD *)a2;
    v10 = v28 + 1;
    if ( 4 * v10 < 3 * v7 )
    {
      if ( v7 - (v10 + *(_DWORD *)(a2 + 20)) > v7 >> 3 )
        goto LABEL_7;
      goto LABEL_6;
    }
LABEL_5:
    v7 *= 2;
LABEL_6:
    sub_15C08C0(a2, v7);
    sub_15B87F0(a2, &v33, &v34);
    v8 = v34;
    v9 = v33;
    v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_7:
    *(_DWORD *)(a2 + 16) = v10;
    if ( *v8 != -8 )
      --*(_DWORD *)(a2 + 20);
    *v8 = v9;
    return v33;
  }
  return result;
}
