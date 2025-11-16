// Function: sub_2E2B9F0
// Address: 0x2e2b9f0
//
__int64 __fastcall sub_2E2B9F0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 result; // rax
  __int64 v8; // r13
  int v11; // ecx
  __int64 v12; // rcx
  _BOOL8 v13; // rdx
  int v14; // esi
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  __int64 v19; // rcx

  v6 = *(_QWORD *)(a2 + 32);
  result = 5LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v8 = v6 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  while ( v8 != v6 )
  {
    if ( *(_BYTE *)v6 )
      goto LABEL_5;
    result = *(unsigned __int8 *)(v6 + 3);
    v11 = result;
    LOBYTE(v11) = (unsigned __int8)result >> 4;
    v12 = v11 ^ 1u;
    v13 = (*(_BYTE *)(v6 + 3) & 0x40) != 0;
    if ( !(v13 & (unsigned __int8)v12) )
      goto LABEL_5;
    v14 = *(_DWORD *)(v6 + 8);
    result = (unsigned int)result & 0xFFFFFFBF;
    *(_BYTE *)(v6 + 3) = result;
    if ( v14 >= 0 )
      goto LABEL_5;
    v15 = sub_2E29D60(a1, v14, v13, v12, a5, a6);
    v16 = *(_QWORD *)(v15 + 40);
    v17 = *(_QWORD *)(v15 + 32);
    v18 = v15;
    result = (v16 - v17) >> 5;
    v19 = (v16 - v17) >> 3;
    if ( result > 0 )
    {
      result = v17 + 32 * result;
      while ( a2 != *(_QWORD *)v17 )
      {
        if ( a2 == *(_QWORD *)(v17 + 8) )
        {
          v17 += 8;
          break;
        }
        if ( a2 == *(_QWORD *)(v17 + 16) )
        {
          v17 += 16;
          break;
        }
        if ( a2 == *(_QWORD *)(v17 + 24) )
        {
          v17 += 24;
          break;
        }
        v17 += 32;
        if ( result == v17 )
        {
          v19 = (v16 - v17) >> 3;
          goto LABEL_17;
        }
      }
LABEL_14:
      if ( v16 != v17 )
        result = (__int64)sub_2E25970(v18 + 32, (_BYTE *)v17);
      goto LABEL_5;
    }
LABEL_17:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_5;
        goto LABEL_20;
      }
      if ( a2 == *(_QWORD *)v17 )
        goto LABEL_14;
      v17 += 8;
    }
    if ( a2 == *(_QWORD *)v17 )
      goto LABEL_14;
    v17 += 8;
LABEL_20:
    if ( a2 == *(_QWORD *)v17 )
      goto LABEL_14;
LABEL_5:
    v6 += 40;
  }
  return result;
}
