// Function: sub_2A63550
// Address: 0x2a63550
//
__int64 __fastcall sub_2A63550(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r13d
  int v5; // r15d
  unsigned __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rdi
  int v9; // r10d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned __int8 v15; // al
  unsigned __int8 v16; // al
  __int64 v18; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+0h] [rbp-70h]
  unsigned __int8 v21; // [rsp+8h] [rbp-68h]
  unsigned __int8 v22; // [rsp+10h] [rbp-60h] BYREF
  char v23; // [rsp+11h] [rbp-5Fh]
  unsigned __int64 v24; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v25; // [rsp+20h] [rbp-50h]
  unsigned __int64 v26; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v27; // [rsp+30h] [rbp-40h]

  v3 = *(_DWORD *)(a3 + 12);
  if ( !v3 )
    return 1;
  v5 = 0;
  v6 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
  while ( 1 )
  {
    v7 = *(unsigned int *)(a1 + 304);
    v8 = *(_QWORD *)(a1 + 288);
    if ( (_DWORD)v7 )
    {
      v9 = 1;
      for ( i = (v7 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v6 | (unsigned int)(37 * v5))) >> 31)
               ^ (484763065 * (v6 | (37 * v5)))); ; i = (v7 - 1) & v12 )
      {
        v11 = v8 + 24LL * i;
        if ( a2 == *(_QWORD *)v11 && *(_DWORD *)(v11 + 8) == v5 )
          break;
        if ( *(_QWORD *)v11 == -4096 && *(_DWORD *)(v11 + 8) == -1 )
          goto LABEL_26;
        v12 = v9 + i;
        ++v9;
      }
      v13 = *(_QWORD *)(a1 + 312);
      if ( v11 != v8 + 24 * v7 )
      {
        v14 = v13 + 56LL * *(unsigned int *)(v11 + 16);
        goto LABEL_11;
      }
    }
    else
    {
LABEL_26:
      v13 = *(_QWORD *)(a1 + 312);
    }
    v14 = v13 + 56LL * *(unsigned int *)(a1 + 320);
LABEL_11:
    v15 = *(_BYTE *)(v14 + 16);
    v23 = 0;
    v22 = v15;
    if ( v15 > 3u )
      break;
    if ( v15 > 1u )
      v24 = *(_QWORD *)(v14 + 24);
LABEL_14:
    v16 = sub_2A62D90((__int64)&v22);
    if ( !v16 )
      goto LABEL_24;
LABEL_15:
    if ( (unsigned int)v22 - 4 <= 1 )
    {
      if ( v27 > 0x40 && v26 )
        j_j___libc_free_0_0(v26);
      if ( v25 > 0x40 )
      {
        if ( v24 )
          j_j___libc_free_0_0(v24);
      }
    }
    if ( ++v5 == v3 )
      return 1;
  }
  if ( (unsigned __int8)(v15 - 4) > 1u )
    goto LABEL_14;
  v25 = *(_DWORD *)(v14 + 32);
  if ( v25 > 0x40 )
  {
    v19 = v14;
    sub_C43780((__int64)&v24, (const void **)(v14 + 24));
    v14 = v19;
  }
  else
  {
    v24 = *(_QWORD *)(v14 + 24);
  }
  v27 = *(_DWORD *)(v14 + 48);
  if ( v27 > 0x40 )
  {
    v18 = v14;
    sub_C43780((__int64)&v26, (const void **)(v14 + 40));
    v14 = v18;
  }
  else
  {
    v26 = *(_QWORD *)(v14 + 40);
  }
  v23 = *(_BYTE *)(v14 + 17);
  v16 = sub_2A62D90((__int64)&v22);
  if ( v16 )
    goto LABEL_15;
LABEL_24:
  v21 = v16;
  sub_22C0090(&v22);
  return v21;
}
