// Function: sub_162B2E0
// Address: 0x162b2e0
//
__int64 __fastcall sub_162B2E0(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v4; // r8d
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 *v7; // r8
  __int64 v8; // rdi
  int v9; // eax
  __int64 result; // rax
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // rsi
  int v14; // r10d
  __int64 v15; // rcx
  __int64 v16; // r13
  int v17; // r13d
  __int64 v18; // rdx
  __int64 *v19; // r9
  __int64 v20; // rsi
  int v21; // r10d
  __int64 *v22; // r11
  int v23; // eax
  int v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v27; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 28);
  v4 = *(_DWORD *)(a2 + 24);
  v26 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  LODWORD(v27) = v3;
  v24 = v4;
  BYTE4(v27) = *(_BYTE *)(a1 + 52);
  v28[0] = *(_QWORD *)(a1 + 8 * (3LL - *(unsigned int *)(a1 + 8)));
  if ( !v4 )
    goto LABEL_2;
  v11 = (v4 - 1) & sub_15B3730((int *)&v27, (__int8 *)&v27 + 4, v28);
  v12 = (__int64 *)(v5 + 8LL * v11);
  v13 = *v12;
  if ( *v12 == -8 )
  {
LABEL_15:
    v15 = *(_QWORD *)(a2 + 8);
    LODWORD(v16) = *(_DWORD *)(a2 + 24);
    goto LABEL_16;
  }
  v14 = 1;
  while ( v13 == -16
       || (_DWORD)v27 != *(_DWORD *)(v13 + 28)
       || BYTE4(v27) != *(_BYTE *)(v13 + 52)
       || v28[0] != *(_QWORD *)(v13 + 8 * (3LL - *(unsigned int *)(v13 + 8))) )
  {
    v11 = (v24 - 1) & (v14 + v11);
    v12 = (__int64 *)(v5 + 8LL * v11);
    v13 = *v12;
    if ( *v12 == -8 )
      goto LABEL_15;
    ++v14;
  }
  v15 = *(_QWORD *)(a2 + 8);
  v16 = *(unsigned int *)(a2 + 24);
  if ( v12 == (__int64 *)(v15 + 8 * v16) || (result = *v12) == 0 )
  {
LABEL_16:
    v25 = v15;
    if ( (_DWORD)v16 )
    {
      v17 = v16 - 1;
      LODWORD(v27) = *(_DWORD *)(v26 + 28);
      BYTE4(v27) = *(_BYTE *)(v26 + 52);
      v28[0] = *(_QWORD *)(v26 + 8 * (3LL - *(unsigned int *)(v26 + 8)));
      v8 = v26;
      LODWORD(v18) = v17 & sub_15B3730((int *)&v27, (__int8 *)&v27 + 4, v28);
      v19 = (__int64 *)(v25 + 8LL * (unsigned int)v18);
      result = v26;
      v20 = *v19;
      if ( *v19 == v26 )
        return result;
      v21 = 1;
      v7 = 0;
      while ( v20 != -8 )
      {
        if ( v20 != -16 || v7 )
          v19 = v7;
        v18 = v17 & (unsigned int)(v18 + v21);
        v22 = (__int64 *)(v25 + 8 * v18);
        v20 = *v22;
        if ( *v22 == v26 )
          return result;
        ++v21;
        v7 = v19;
        v19 = (__int64 *)(v25 + 8 * v18);
      }
      v23 = *(_DWORD *)(a2 + 16);
      v6 = *(_DWORD *)(a2 + 24);
      if ( !v7 )
        v7 = v19;
      ++*(_QWORD *)a2;
      v9 = v23 + 1;
      if ( 4 * v9 < 3 * v6 )
      {
        if ( v6 - (v9 + *(_DWORD *)(a2 + 20)) > v6 >> 3 )
          goto LABEL_5;
        goto LABEL_4;
      }
LABEL_3:
      v6 *= 2;
LABEL_4:
      sub_15BEB90(a2, v6);
      sub_15B80C0(a2, &v26, &v27);
      v7 = v27;
      v8 = v26;
      v9 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
      *(_DWORD *)(a2 + 16) = v9;
      if ( *v7 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v7 = v8;
      return v26;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v6 = 0;
    goto LABEL_3;
  }
  return result;
}
