// Function: sub_1EA4010
// Address: 0x1ea4010
//
unsigned int *__fastcall sub_1EA4010(unsigned int *a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // rbx
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 i; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  bool v24; // [rsp+18h] [rbp-48h]
  _QWORD v25[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = sub_1E69D00(a2, a3);
  v5 = *(_QWORD *)(v4 + 24);
  v6 = *(_QWORD *)(v4 + 32);
  v7 = *(_QWORD *)(v5 + 32);
  *(_QWORD *)a1 = a1 + 4;
  *((_QWORD *)a1 + 1) = 0x400000000LL;
  v8 = *(_DWORD *)(v4 + 40);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v4 + 32);
    v10 = v9 + 40LL * (unsigned int)(v8 - 1) + 40;
    while ( *(_BYTE *)v9 || (*(_BYTE *)(v9 + 3) & 0x10) == 0 || a3 != *(_DWORD *)(v9 + 8) )
    {
      v9 += 40;
      if ( v10 == v9 )
        goto LABEL_7;
    }
    v6 = v9;
  }
  else
  {
LABEL_7:
    if ( !v6 )
      return a1;
  }
  for ( i = v5 + 24; i != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    while ( 1 )
    {
      if ( **(_WORD **)(v7 + 16) == 12 )
      {
        v13 = *(_QWORD *)(v7 + 32);
        if ( !*(_BYTE *)v13 && *(_DWORD *)(v6 + 8) == *(_DWORD *)(v13 + 8) )
        {
          v25[0] = 0;
          v14 = *(_QWORD *)(v7 + 32);
          v24 = *(_BYTE *)(v14 + 40) == 1;
          if ( *(_BYTE *)(v14 + 40) == 1 )
            v15 = *(_QWORD *)(v14 + 64);
          else
            v15 = *(unsigned int *)(v14 + 48);
          v21 = v15;
          v22 = *(_QWORD *)(v14 + 104);
          v23 = *(_QWORD *)(v14 + 144);
          if ( v25 != (_QWORD *)(v7 + 64) )
          {
            v16 = *(_QWORD *)(v7 + 64);
            v25[0] = v16;
            if ( v16 )
              sub_1623A60((__int64)v25, v16, 2);
          }
          v17 = a1[2];
          if ( v17 >= a1[3] )
          {
            sub_1EA3E60(a1, 0);
            v17 = a1[2];
          }
          v18 = *(_QWORD *)a1 + 40LL * v17;
          if ( v18 )
          {
            *(_QWORD *)v18 = v21;
            *(_QWORD *)(v18 + 8) = v22;
            *(_QWORD *)(v18 + 16) = v23;
            *(_BYTE *)(v18 + 24) = v24;
            v19 = v25[0];
            *(_QWORD *)(v18 + 32) = v25[0];
            if ( v19 )
              sub_1623A60(v18 + 32, v19, 2);
            v17 = a1[2];
          }
          v20 = v25[0];
          a1[2] = v17 + 1;
          if ( v20 )
            sub_161E7C0((__int64)v25, v20);
        }
      }
      if ( (*(_BYTE *)v7 & 4) == 0 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( i == v7 )
        return a1;
    }
    while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
      v7 = *(_QWORD *)(v7 + 8);
  }
  return a1;
}
