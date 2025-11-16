// Function: sub_1DC29C0
// Address: 0x1dc29c0
//
__int64 __fastcall sub_1DC29C0(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 *v8; // r13
  __int64 *i; // rbx
  __int64 v10; // rsi
  __int64 result; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int16 v14; // ax
  __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 j; // r13
  unsigned int v18; // esi

  v8 = (__int64 *)a2[12];
  for ( i = (__int64 *)a2[11]; v8 != i; ++i )
  {
    v10 = *i;
    sub_1DC26F0(a1, v10);
  }
  result = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  v12 = result;
  if ( (_QWORD *)result == a2 + 3 )
    return result;
  if ( !result )
    BUG();
  v13 = *(_QWORD *)result;
  v14 = *(_WORD *)(result + 46);
  v15 = v14 & 4;
  if ( (v13 & 4) != 0 )
  {
    if ( (v14 & 4) != 0 )
      goto LABEL_7;
  }
  else if ( (v14 & 4) != 0 )
  {
    while ( 1 )
    {
      v13 &= 0xFFFFFFFFFFFFFFF8LL;
      v14 = *(_WORD *)(v13 + 46);
      v12 = v13;
      if ( (v14 & 4) == 0 )
        break;
      v13 = *(_QWORD *)v13;
    }
  }
  if ( (v14 & 8) != 0 )
  {
    result = sub_1E15D00(v12, 8, 1);
    goto LABEL_8;
  }
LABEL_7:
  result = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 8LL) >> 3) & 1LL;
LABEL_8:
  if ( (_BYTE)result )
  {
    result = *(_QWORD *)(a2[7] + 56LL);
    if ( *(_BYTE *)(result + 104) )
    {
      v16 = *(_QWORD *)(result + 80);
      for ( j = *(_QWORD *)(result + 88); j != v16; result = sub_1DC1BF0(a1, v18, v13, v15, a5, a6) )
      {
        while ( !*(_BYTE *)(v16 + 8) )
        {
          v16 += 12;
          if ( j == v16 )
            return result;
        }
        v18 = *(_DWORD *)v16;
        v16 += 12;
      }
    }
  }
  return result;
}
