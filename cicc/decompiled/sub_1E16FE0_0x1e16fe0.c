// Function: sub_1E16FE0
// Address: 0x1e16fe0
//
__int64 __fastcall sub_1E16FE0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  unsigned int v9; // ebx
  unsigned int v10; // r12d
  __int64 v11; // rsi
  __int64 v12; // rax
  unsigned __int64 v14; // rbx
  __int64 v15; // r10
  __int64 v16; // r11
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-40h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  if ( a6 )
  {
    v14 = a1;
    if ( (*(_BYTE *)(a1 + 46) & 4) != 0 )
    {
      do
        v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v14 + 46) & 4) != 0 );
    }
    v15 = *(_QWORD *)(a1 + 24) + 24LL;
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 32);
      v17 = v16 + 40LL * *(unsigned int *)(v14 + 40);
      if ( v16 != v17 )
        break;
      v14 = *(_QWORD *)(v14 + 8);
      if ( v15 == v14 || (*(_BYTE *)(v14 + 46) & 4) == 0 )
      {
        if ( v17 == v16 )
          return a3;
        break;
      }
    }
    while ( a3 )
    {
      v20 = v15;
      v21 = v16;
      v18 = sub_1E16FA0(
              *(_QWORD *)(v16 + 16),
              0xCCCCCCCCCCCCCCCDLL * ((v16 - *(_QWORD *)(v14 + 32)) >> 3),
              a2,
              a3,
              a4,
              a5);
      v15 = v20;
      a3 = v18;
      v19 = v17;
      if ( v21 + 40 == v17 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(v14 + 8);
          if ( v20 == v14 || (*(_BYTE *)(v14 + 46) & 4) == 0 )
            break;
          v17 = *(_QWORD *)(v14 + 32);
          v19 = v17 + 40LL * *(unsigned int *)(v14 + 40);
          if ( v17 != v19 )
            goto LABEL_21;
        }
        if ( v17 == v19 )
          return a3;
      }
      else
      {
        v17 = v21 + 40;
      }
LABEL_21:
      v16 = v17;
      v17 = v19;
    }
    return 0;
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 40);
    if ( v9 && a3 )
    {
      v10 = 0;
      do
      {
        v11 = v10++;
        v12 = sub_1E16FA0(a1, v11, a2, a3, a4, a5);
        a3 = v12;
      }
      while ( v9 > v10 && v12 );
    }
  }
  return a3;
}
