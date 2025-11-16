// Function: sub_162C2C0
// Address: 0x162c2c0
//
__int64 __fastcall sub_162C2C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // r15d
  __int64 v5; // r13
  unsigned int v6; // r8d
  int v7; // esi
  __int64 *v8; // rcx
  __int64 v9; // rdi
  int v10; // eax
  __int64 result; // rax
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // r13
  int v18; // r13d
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 *v21; // rsi
  int v22; // r8d
  __int64 *v23; // r9
  int v24; // eax
  int i; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v27; // [rsp+10h] [rbp-50h] BYREF
  __int64 v28; // [rsp+18h] [rbp-48h] BYREF
  char v29; // [rsp+20h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_DWORD *)(a2 + 24);
  v26 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v27 = *(__int64 **)(a1 + 8 * (1 - v3));
  v28 = *(_QWORD *)(a1 + 8 * (2 - v3));
  v29 = *(_BYTE *)(a1 + 24) & 1;
  if ( !v4 )
    goto LABEL_2;
  v12 = (v4 - 1) & sub_15B2420((__int64 *)&v27, &v28);
  v13 = (__int64 *)(v5 + 8LL * v12);
  v14 = *v13;
  if ( *v13 == -8 )
  {
LABEL_15:
    v16 = *(_QWORD *)(a2 + 8);
    LODWORD(v17) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  for ( i = 1; ; ++i )
  {
    if ( v14 != -16 )
    {
      v15 = *(unsigned int *)(v14 + 8);
      if ( v27 == *(__int64 **)(v14 + 8 * (1 - v15))
        && v28 == *(_QWORD *)(v14 + 8 * (2 - v15))
        && v29 == (*(_BYTE *)(v14 + 24) & 1) )
      {
        break;
      }
    }
    v12 = (v4 - 1) & (i + v12);
    v13 = (__int64 *)(v5 + 8LL * v12);
    v14 = *v13;
    if ( *v13 == -8 )
      goto LABEL_15;
  }
  v16 = *(_QWORD *)(a2 + 8);
  v17 = *(unsigned int *)(a2 + 24);
  if ( v13 == (__int64 *)(v16 + 8 * v17) || (result = *v13) == 0 )
  {
LABEL_16:
    if ( (_DWORD)v17 )
    {
      v18 = v17 - 1;
      v19 = *(unsigned int *)(v26 + 8);
      v27 = *(__int64 **)(v26 + 8 * (1 - v19));
      v28 = *(_QWORD *)(v26 + 8 * (2 - v19));
      v29 = *(_BYTE *)(v26 + 24) & 1;
      v9 = v26;
      LODWORD(v20) = v18 & sub_15B2420((__int64 *)&v27, &v28);
      v21 = (__int64 *)(v16 + 8LL * (unsigned int)v20);
      result = *v21;
      if ( v26 == *v21 )
        return result;
      v22 = 1;
      v8 = 0;
      while ( result != -8 )
      {
        if ( result != -16 || v8 )
          v21 = v8;
        v20 = v18 & (unsigned int)(v20 + v22);
        v23 = (__int64 *)(v16 + 8 * v20);
        result = *v23;
        if ( *v23 == v26 )
          return result;
        ++v22;
        v8 = v21;
        v21 = (__int64 *)(v16 + 8 * v20);
      }
      v24 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v8 )
        v8 = v21;
      ++*(_QWORD *)a2;
      v10 = v24 + 1;
      if ( 4 * v10 < 3 * v6 )
      {
        if ( v6 - (v10 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        v7 = v6;
LABEL_4:
        sub_15C0E60(a2, v7);
        sub_15B88F0(a2, &v26, &v27);
        v8 = v27;
        v9 = v26;
        v10 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v10;
        if ( *v8 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v8 = v9;
        return v26;
      }
LABEL_3:
      v7 = 2 * v6;
      goto LABEL_4;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v6 = 0;
    goto LABEL_3;
  }
  return result;
}
