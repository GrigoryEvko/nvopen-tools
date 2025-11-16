// Function: sub_D541A0
// Address: 0xd541a0
//
__int64 __fastcall sub_D541A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 v5; // r9
  __int64 v6; // r10
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  char v10; // cl
  char v12; // cl
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rcx
  __int64 v16; // rcx

  v3 = a2 - a1;
  v5 = v3 >> 5;
  if ( v3 <= 0 )
    return 1;
  while ( 1 )
  {
    v6 = *a3;
    v7 = a1;
    v8 = (a3[2] - *a3) >> 5;
    if ( v8 > v5 )
      v8 = v5;
    a1 += 32 * v8;
    if ( a1 != v7 )
      break;
LABEL_14:
    v5 -= v8;
    v13 = v8 + ((v6 - a3[1]) >> 5);
    if ( v13 < 0 )
    {
      v14 = ~((unsigned __int64)~v13 >> 4);
      goto LABEL_20;
    }
    if ( v13 > 15 )
    {
      v14 = v13 >> 4;
LABEL_20:
      v15 = (__int64 *)(a3[3] + 8 * v14);
      a3[3] = (__int64)v15;
      v16 = *v15;
      a3[1] = v16;
      a3[2] = v16 + 512;
      *a3 = v16 + 32 * (v13 - 16 * v14);
      goto LABEL_17;
    }
    *a3 = 32 * v8 + v6;
LABEL_17:
    if ( v5 <= 0 )
      return 1;
  }
  v9 = *a3;
  while ( 1 )
  {
    v10 = *(_BYTE *)(v7 + 24);
    if ( v10 != *(_BYTE *)(v9 + 24) )
      return 0;
    if ( v10 )
    {
      if ( *(_QWORD *)v7 != *(_QWORD *)v9 )
        return 0;
      v12 = *(_BYTE *)(v7 + 16);
      if ( v12 != *(_BYTE *)(v9 + 16) || v12 && *(_QWORD *)(v7 + 8) != *(_QWORD *)(v9 + 8) )
        return 0;
    }
    v7 += 32;
    v9 += 32;
    if ( a1 == v7 )
      goto LABEL_14;
  }
}
