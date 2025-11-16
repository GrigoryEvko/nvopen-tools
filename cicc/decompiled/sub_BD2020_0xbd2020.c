// Function: sub_BD2020
// Address: 0xbd2020
//
char __fastcall sub_BD2020(__int64 a1, __int64 a2)
{
  __int64 v2; // r11
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // ebx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rcx
  __int64 v10; // r8
  int v11; // eax
  int v12; // edx
  __int64 *v13; // r13
  __int64 *v14; // r15
  __int64 *j; // rbx
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r8d
  __int64 v19; // r11
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // r9d
  _QWORD *v23; // rsi
  int v24; // r8d
  int v25; // r8d
  __int64 v26; // r11
  int v27; // r9d
  unsigned int v28; // eax
  __int64 v29; // rdi
  __int64 *i; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h] BYREF
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  __int64 v34[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = a1 + 64;
  v32 = a2;
  v4 = *(_DWORD *)(a1 + 88);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_25;
  }
  v5 = *(_QWORD *)(a1 + 72);
  v6 = 1;
  LODWORD(v7) = (v4 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  v8 = (_QWORD *)(v5 + 8LL * (unsigned int)v7);
  v9 = 0;
  v10 = *v8;
  if ( v32 == *v8 )
    return v7;
  while ( v10 != -4 )
  {
    if ( !v9 && v10 == -8 )
      v9 = v8;
    LODWORD(v7) = (v4 - 1) & (v6 + v7);
    v8 = (_QWORD *)(v5 + 8LL * (unsigned int)v7);
    v10 = *v8;
    if ( v32 == *v8 )
      return v7;
    ++v6;
  }
  v11 = *(_DWORD *)(a1 + 80);
  if ( !v9 )
    v9 = v8;
  ++*(_QWORD *)(a1 + 64);
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v4 )
  {
LABEL_25:
    sub_BD1E50(v2, 2 * v4);
    v17 = *(_DWORD *)(a1 + 88);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 72);
      v20 = v18 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v9 = (_QWORD *)(v19 + 8LL * v20);
      v21 = *v9;
      v12 = *(_DWORD *)(a1 + 80) + 1;
      if ( *v9 == v32 )
        goto LABEL_14;
      v22 = 1;
      v23 = 0;
      while ( v21 != -4 )
      {
        if ( v21 == -8 && !v23 )
          v23 = v9;
        v20 = v18 & (v22 + v20);
        v9 = (_QWORD *)(v19 + 8LL * v20);
        v21 = *v9;
        if ( v32 == *v9 )
          goto LABEL_14;
        ++v22;
      }
      goto LABEL_29;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 84) - v12 > v4 >> 3 )
    goto LABEL_14;
  sub_BD1E50(v2, v4);
  v24 = *(_DWORD *)(a1 + 88);
  if ( !v24 )
    goto LABEL_45;
  v25 = v24 - 1;
  v26 = *(_QWORD *)(a1 + 72);
  v23 = 0;
  v27 = 1;
  v28 = v25 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  v9 = (_QWORD *)(v26 + 8LL * v28);
  v29 = *v9;
  v12 = *(_DWORD *)(a1 + 80) + 1;
  if ( v32 == *v9 )
    goto LABEL_14;
  while ( v29 != -4 )
  {
    if ( v29 == -8 && !v23 )
      v23 = v9;
    v28 = v25 & (v27 + v28);
    v9 = (_QWORD *)(v26 + 8LL * v28);
    v29 = *v9;
    if ( v32 == *v9 )
      goto LABEL_14;
    ++v27;
  }
LABEL_29:
  if ( v23 )
    v9 = v23;
LABEL_14:
  *(_DWORD *)(a1 + 80) = v12;
  if ( *v9 != -4 )
    --*(_DWORD *)(a1 + 84);
  *v9 = v32;
  v13 = (__int64 *)sub_A74450(&v32);
  v7 = sub_A74460(&v32);
  for ( i = (__int64 *)v7; i != v13; ++v13 )
  {
    v33 = *v13;
    v14 = (__int64 *)sub_A73280(&v33);
    v7 = sub_A73290(&v33);
    for ( j = (__int64 *)v7; j != v14; LOBYTE(v7) = sub_BD0F10(a1, v16) )
    {
      while ( 1 )
      {
        v34[0] = *v14;
        LOBYTE(v7) = sub_A71860((__int64)v34);
        if ( (_BYTE)v7 )
          break;
        if ( j == ++v14 )
          goto LABEL_22;
      }
      ++v14;
      v16 = sub_A72A60(v34);
    }
LABEL_22:
    ;
  }
  return v7;
}
