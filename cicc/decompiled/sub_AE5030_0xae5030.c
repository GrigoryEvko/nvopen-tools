// Function: sub_AE5030
// Address: 0xae5030
//
__int64 __fastcall sub_AE5030(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  char v5; // cl
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 result; // rax
  char v9; // dl
  __int64 v10; // r14
  unsigned __int8 v11; // cl
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  char v16; // cl
  __int64 v17; // [rsp+0h] [rbp-50h]
  char v18; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v19; // [rsp+Fh] [rbp-41h]

  *(_WORD *)(a1 + 16) &= 0xFE00u;
  v4 = *(_DWORD *)(a2 + 12);
  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 20) = v4;
  *(_BYTE *)(a1 + 22) = BYTE2(v4);
  *(_BYTE *)(a1 + 8) = 0;
  v5 = HIBYTE(v4) & 0x7F;
  v4 &= ~0x80000000;
  *(_BYTE *)(a1 + 23) = v5 | *(_BYTE *)(a1 + 23) & 0x80;
  v17 = v4;
  if ( v4 )
  {
    v6 = 0;
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v6);
      if ( !v6 && (unsigned __int8)sub_BCEA30(v10) )
      {
        *(_QWORD *)a1 = 0;
        *(_BYTE *)(a1 + 8) = 1;
      }
      v11 = 0;
      if ( (*(_BYTE *)(a2 + 9) & 2) != 0 )
      {
        if ( !*(_BYTE *)(a1 + 8) )
          goto LABEL_12;
      }
      else
      {
        v11 = sub_AE5020(a3, v10);
        if ( !*(_BYTE *)(a1 + 8) )
        {
LABEL_12:
          v19 = v11;
          v12 = sub_CA1930(a1);
          v11 = v19;
          if ( (v12 & ~(-1LL << v19)) != 0 )
          {
            *(_BYTE *)(a1 + 17) |= 1u;
            v13 = sub_CA1930(a1);
            v11 = v19;
            *(_BYTE *)(a1 + 8) = 0;
            *(_QWORD *)a1 = ((1LL << v19) + v13 - 1) & -(1LL << v19);
          }
        }
      }
      if ( v11 >= *(_BYTE *)(a1 + 16) )
        *(_BYTE *)(a1 + 16) = v11;
      v7 = 16 * v6;
      *(_QWORD *)(a1 + v7 + 24) = *(_QWORD *)a1;
      *(_BYTE *)(a1 + v7 + 32) = *(_BYTE *)(a1 + 8);
      v18 = sub_AE5020(a3, v10);
      result = (((unsigned __int64)(sub_9208B0(a3, v10) + 7) >> 3) + (1LL << v18) - 1) >> v18 << v18;
      *(_QWORD *)a1 += result;
      if ( result )
        *(_BYTE *)(a1 + 8) = v9;
      if ( v17 == ++v6 )
      {
        if ( *(_BYTE *)(a1 + 8) )
          return result;
        break;
      }
    }
  }
  v14 = sub_CA1930(a1);
  result = ~(-1LL << *(_BYTE *)(a1 + 16));
  if ( (v14 & result) != 0 )
  {
    *(_BYTE *)(a1 + 17) |= 1u;
    v15 = sub_CA1930(a1);
    v16 = *(_BYTE *)(a1 + 16);
    *(_BYTE *)(a1 + 8) = 0;
    result = ((1LL << v16) + v15 - 1) & -(1LL << v16);
    *(_QWORD *)a1 = result;
  }
  return result;
}
