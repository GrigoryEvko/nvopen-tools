// Function: sub_1361B70
// Address: 0x1361b70
//
__int64 __fastcall sub_1361B70(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r14d
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // r11
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r10
  __int64 v13; // rbx
  int v14; // r15d
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  unsigned int i; // eax
  __int64 *v18; // rdx
  __int64 v19; // r15
  unsigned int v20; // eax
  int v21; // edx
  __int64 v23; // [rsp+0h] [rbp-50h]
  int v24; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+20h] [rbp-30h]

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = 7;
    v27 = a1 + 16;
  }
  else
  {
    v21 = *(_DWORD *)(a1 + 24);
    v27 = *(_QWORD *)(a1 + 16);
    v4 = v21 - 1;
    if ( !v21 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v5 = *a2;
  v24 = 1;
  v6 = a2[1];
  v7 = a2[2];
  v25 = 0;
  v8 = a2[3];
  v9 = a2[4];
  v23 = a2[9];
  v10 = a2[5];
  v11 = a2[6];
  v12 = a2[7];
  v13 = a2[8];
  v14 = ((unsigned int)v13 >> 9)
      ^ ((unsigned int)v12 >> 9)
      ^ (37 * v11)
      ^ ((unsigned int)v10 >> 4)
      ^ ((unsigned int)v10 >> 9)
      ^ ((unsigned int)v12 >> 4)
      ^ ((unsigned int)v13 >> 4)
      ^ ((unsigned int)v23 >> 4);
  v15 = (((v14 ^ ((unsigned int)v23 >> 9)
         | ((unsigned __int64)(((unsigned int)v9 >> 9)
                             ^ ((unsigned int)v8 >> 9)
                             ^ ((unsigned int)v7 >> 9)
                             ^ (37 * (_DWORD)v6)
                             ^ ((unsigned int)v5 >> 9)
                             ^ ((unsigned int)v5 >> 4)
                             ^ ((unsigned int)v7 >> 4)
                             ^ ((unsigned int)v8 >> 4)
                             ^ ((unsigned int)v9 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v14 ^ ((unsigned int)v23 >> 9)) << 32)) >> 22)
      ^ ((v14 ^ ((unsigned int)v23 >> 9)
        | ((unsigned __int64)(((unsigned int)v9 >> 9)
                            ^ ((unsigned int)v8 >> 9)
                            ^ ((unsigned int)v7 >> 9)
                            ^ (37 * (_DWORD)v6)
                            ^ ((unsigned int)v5 >> 9)
                            ^ ((unsigned int)v5 >> 4)
                            ^ ((unsigned int)v7 >> 4)
                            ^ ((unsigned int)v8 >> 4)
                            ^ ((unsigned int)v9 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v14 ^ ((unsigned int)v23 >> 9)) << 32));
  v16 = 9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13)));
  for ( i = v4
          & ((((v16 ^ (v16 >> 15)) - 1 - ((v16 ^ (v16 >> 15)) << 27)) >> 31)
           ^ ((v16 ^ (v16 >> 15)) - 1 - (((unsigned int)v16 ^ (unsigned int)(v16 >> 15)) << 27))); ; i = v4 & v20 )
  {
    v18 = (__int64 *)(v27 + 88LL * i);
    v19 = *v18;
    if ( v5 == *v18
      && v6 == v18[1]
      && v7 == v18[2]
      && v8 == v18[3]
      && v9 == v18[4]
      && v10 == v18[5]
      && v11 == v18[6]
      && v12 == v18[7]
      && v13 == v18[8]
      && v23 == v18[9] )
    {
      *a3 = v18;
      return 1;
    }
    if ( v19 == -8 )
      break;
    if ( v19 == -16 && !v18[1] && !v18[2] && !v18[3] && !v18[4] && v18[5] == -16 && !v18[6] && !v18[7] && !v18[8] )
    {
      if ( v18[9] | v25 )
        v18 = (__int64 *)v25;
      v25 = (unsigned __int64)v18;
    }
LABEL_8:
    v20 = v24 + i;
    ++v24;
  }
  if ( v18[1] || v18[2] || v18[3] || v18[4] || v18[5] != -8 || v18[6] || v18[7] || v18[8] || v18[9] )
    goto LABEL_8;
  if ( v25 )
    v18 = (__int64 *)v25;
  *a3 = v18;
  return 0;
}
