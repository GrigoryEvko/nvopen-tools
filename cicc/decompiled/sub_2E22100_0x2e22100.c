// Function: sub_2E22100
// Address: 0x2e22100
//
__int64 __fastcall sub_2E22100(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // rbx
  __int64 result; // rax
  unsigned __int8 *v4; // r13
  unsigned int v5; // eax
  __int16 v6; // cx
  unsigned int v7; // ecx
  __int64 v8; // rdx
  __int64 v9; // rsi

  v2 = *(unsigned __int8 **)(a2 + 32);
  result = 5LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v4 = &v2[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v4 != v2 )
  {
    while ( 1 )
    {
      result = *v2;
      if ( !(_BYTE)result )
        break;
      if ( (_BYTE)result == 12 )
      {
        v9 = *((_QWORD *)v2 + 3);
        v2 += 40;
        result = sub_2E21EB0(a1, v9);
        if ( v4 == v2 )
          return result;
      }
      else
      {
LABEL_10:
        v2 += 40;
        if ( v4 == v2 )
          return result;
      }
    }
    result = *((unsigned int *)v2 + 2);
    if ( (unsigned int)(result - 1) <= 0x3FFFFFFE && ((v2[3] & 0x10) != 0 || (v2[4] & 1) == 0 && (v2[4] & 2) == 0) )
    {
      v5 = *(_DWORD *)(*(_QWORD *)(*a1 + 8) + 24 * result + 16);
      v6 = v5;
      result = v5 >> 12;
      v7 = v6 & 0xFFF;
      v8 = *(_QWORD *)(*a1 + 56) + 2 * result;
      do
      {
        if ( !v8 )
          break;
        v8 += 2;
        result = v7 >> 6;
        *(_QWORD *)(a1[1] + 8 * result) |= 1LL << v7;
        v7 += *(__int16 *)(v8 - 2);
      }
      while ( *(_WORD *)(v8 - 2) );
    }
    goto LABEL_10;
  }
  return result;
}
