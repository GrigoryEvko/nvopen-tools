// Function: sub_13966D0
// Address: 0x13966d0
//
__int64 __fastcall sub_13966D0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int8 v6; // al
  __int64 v7; // rsi
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  int v15; // r10d
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rax
  int v19; // edi
  unsigned int i; // eax
  __int64 v21; // r9
  unsigned int v22; // eax
  unsigned __int8 v23; // al
  __int64 v24; // r8
  __int64 v25; // rsi
  int v26; // r11d
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // r9
  unsigned int j; // eax
  __int64 v30; // r9
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // r12
  __int64 v36; // r13

  v3 = *a2;
  if ( *(_BYTE *)(*(_QWORD *)*a2 + 8LL) != 15 )
    return 0;
  v4 = *a3;
  if ( *(_BYTE *)(*(_QWORD *)*a3 + 8LL) != 15 )
    return 0;
  v6 = *(_BYTE *)(v3 + 16);
  if ( v6 > 0x17u )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + 56LL);
    goto LABEL_6;
  }
  if ( v6 == 17 )
  {
    v7 = *(_QWORD *)(v3 + 24);
LABEL_6:
    v8 = *(_BYTE *)(v4 + 16);
    if ( v8 <= 0x17u )
    {
      if ( v8 == 17 )
      {
        v9 = *(_QWORD *)(v4 + 24);
        v10 = v9 | v7;
      }
      else
      {
        v10 = v7;
        v9 = 0;
      }
LABEL_8:
      if ( !v10 )
        return 1;
      if ( v7 )
        goto LABEL_10;
      goto LABEL_24;
    }
LABEL_7:
    v9 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 56LL);
    v10 = v7 | v9;
    goto LABEL_8;
  }
  v23 = *(_BYTE *)(v4 + 16);
  v7 = 0;
  if ( v23 > 0x17u )
    goto LABEL_7;
  if ( v23 != 17 )
    return 1;
  v9 = *(_QWORD *)(v4 + 24);
  if ( !v9 )
    return 1;
LABEL_24:
  v7 = v9;
LABEL_10:
  v11 = sub_1395F70(a1, v7);
  v12 = *((unsigned int *)v11 + 6);
  v13 = v11;
  if ( !(_DWORD)v12 )
    return 1;
  v14 = v11[1];
  v15 = 1;
  v16 = ((((unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32) - 1) >> 22)
      ^ (((unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32) - 1);
  v17 = ((9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13)))) >> 15)
      ^ (9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13))));
  v18 = v17 - 1 - (v17 << 27);
  v19 = v12 - 1;
  for ( i = (v12 - 1) & ((v18 >> 31) ^ v18); ; i = v19 & v22 )
  {
    v21 = v14 + 24LL * i;
    if ( v3 == *(_QWORD *)v21 && !*(_DWORD *)(v21 + 8) )
      break;
    if ( *(_QWORD *)v21 == -8 && *(_DWORD *)(v21 + 8) == -1 )
      return 1;
    v22 = v15 + i;
    ++v15;
  }
  v24 = v14 + 24 * v12;
  if ( v24 == v21 )
    return 1;
  v25 = *(unsigned int *)(v21 + 16);
  v26 = 1;
  v27 = ((((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32) - 1) >> 22)
      ^ (((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32) - 1);
  v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
      ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
  for ( j = v19 & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; j = v19 & v31 )
  {
    v30 = v14 + 24LL * j;
    if ( v4 == *(_QWORD *)v30 && !*(_DWORD *)(v30 + 8) )
      break;
    if ( *(_QWORD *)v30 == -8 && *(_DWORD *)(v30 + 8) == -1 )
      return 1;
    v31 = v26 + j;
    ++v26;
  }
  if ( v24 == v30 )
    return 1;
  v32 = *(unsigned int *)(v30 + 16);
  if ( (_DWORD)v32 == (_DWORD)v25 )
    return 1;
  v33 = v13[4];
  v34 = 16 * v25;
  v35 = *(_QWORD *)(v33 + v34 + 8);
  v36 = *(_QWORD *)(v33 + 16 * v32 + 8);
  if ( !v35 || !v36 )
    return 0;
  if ( (unsigned __int8)sub_14C8180(*(_QWORD *)(v33 + v34 + 8)) || (unsigned __int8)sub_14C8180(v36) )
    return 1;
  if ( !(unsigned __int8)sub_14C8210(v35) )
    return 0;
  return sub_14C8210(v36);
}
