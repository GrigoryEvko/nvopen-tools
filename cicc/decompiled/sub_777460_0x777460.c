// Function: sub_777460
// Address: 0x777460
//
unsigned __int64 __fastcall sub_777460(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v6; // rbx
  unsigned __int64 v7; // rcx
  char v8; // al
  unsigned __int64 v9; // rsi
  char i; // di
  unsigned int j; // edx
  __int64 v12; // rax
  char v14; // al
  unsigned int v15; // esi
  unsigned int v16; // eax
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
    return 0;
  v3 = *(_BYTE *)(a2 + 8);
  if ( (v3 & 2) != 0 )
    return 0;
  if ( (v3 & 8) != 0 )
  {
    v2 = *(_QWORD *)(a2 + 16);
    if ( (v3 & 4) != 0 )
      v2 = *(_QWORD *)(v2 + 24);
  }
  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(_QWORD *)(v6 - 8);
  if ( v6 == v2 )
  {
    if ( *(_BYTE *)(v7 + 140) == 8 && (*(_BYTE *)(v2 - 9) & 2) == 0 )
    {
      do
      {
        v7 = *(_QWORD *)(v7 + 160);
        if ( *(_BYTE *)(v7 + 140) != 12 )
          break;
        v7 = *(_QWORD *)(v7 + 160);
      }
      while ( *(_BYTE *)(v7 + 140) == 12 );
    }
  }
  else
  {
    do
    {
      v8 = *(_BYTE *)(v7 + 140);
      if ( v8 == 8 )
      {
        LODWORD(v19[0]) = 1;
        do
        {
          v7 = *(_QWORD *)(v7 + 160);
          v14 = *(_BYTE *)(v7 + 140);
        }
        while ( v14 == 12 );
        v15 = 16;
        if ( (unsigned __int8)(v14 - 2) > 1u )
        {
          v17 = v7;
          v16 = sub_7764B0(a1, v7, v19);
          v7 = v17;
          v15 = v16;
        }
        i = *(_BYTE *)(v7 + 140);
        v6 += v15 * (((int)v2 - (int)v6) / v15);
      }
      else
      {
        if ( (unsigned __int8)(v8 - 9) > 2u )
          sub_721090();
        sub_777100(a1, a2, v6, v7, &v18, v19);
        v9 = v18;
        if ( v18 )
        {
          v7 = *(_QWORD *)(v18 + 120);
          for ( i = *(_BYTE *)(v7 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
            v7 = *(_QWORD *)(v7 + 160);
        }
        else
        {
          v9 = v19[0];
          v7 = *(_QWORD *)(v19[0] + 40);
          i = *(_BYTE *)(v7 + 140);
        }
        for ( j = qword_4F08388 & (v9 >> 3); ; j = qword_4F08388 & (j + 1) )
        {
          v12 = qword_4F08380 + 16LL * j;
          if ( v9 == *(_QWORD *)v12 )
            break;
          if ( !*(_QWORD *)v12 )
            goto LABEL_16;
        }
        v6 += *(unsigned int *)(v12 + 8);
      }
LABEL_16:
      if ( i == 8 )
      {
        do
        {
          v7 = *(_QWORD *)(v7 + 160);
          if ( *(_BYTE *)(v7 + 140) != 12 )
            break;
          v7 = *(_QWORD *)(v7 + 160);
        }
        while ( *(_BYTE *)(v7 + 140) == 12 );
      }
    }
    while ( v6 != v2 );
  }
  return v7;
}
