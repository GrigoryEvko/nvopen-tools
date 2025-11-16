// Function: sub_12F9B10
// Address: 0x12f9b10
//
bool __fastcall sub_12F9B10(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // rdx
  bool result; // al

  v2 = a2[1];
  v3 = *a2;
  v4 = a1[1];
  v5 = v2;
  v6 = *a1;
  v7 = v3;
  if ( (~v4 & 0x7FFF000000000000LL) == 0 )
  {
    if ( v6 )
    {
      v8 = 0x7FFF800000000000LL;
      v4 &= 0x7FFF800000000000uLL;
      if ( v4 == 0x7FFF000000000000LL )
        goto LABEL_15;
      goto LABEL_9;
    }
    if ( (v4 & 0xFFFFFFFFFFFFLL) != 0 )
    {
      v8 = v4 & 0x7FFF800000000000LL;
      if ( (v4 & 0x7FFF800000000000LL) != 0x7FFF000000000000LL )
        goto LABEL_9;
      goto LABEL_17;
    }
  }
  v8 = 0x7FFF000000000000LL;
  if ( (~v5 & 0x7FFF000000000000LL) == 0 && v3 | v5 & 0xFFFFFFFFFFFFLL )
  {
    if ( (v4 & 0x7FFF800000000000LL) != 0x7FFF000000000000LL )
    {
LABEL_9:
      result = 0;
      v4 = 0x7FFF000000000000LL;
      if ( (v5 & 0x7FFF800000000000LL) != 0x7FFF000000000000LL )
        return result;
      v8 = 0x7FFFFFFFFFFFLL;
      v5 = v7 | v5 & 0x7FFFFFFFFFFFLL;
      if ( !v5 )
        return result;
LABEL_15:
      sub_12F9B70(16, v4, v8, v5, v7);
      return 0;
    }
    if ( v6 )
      goto LABEL_15;
LABEL_17:
    if ( (v4 & 0x7FFFFFFFFFFFLL) != 0 )
      goto LABEL_15;
    goto LABEL_9;
  }
  result = 0;
  if ( v6 == v7 )
  {
    result = 1;
    if ( v4 != v5 )
      return (v6 | (v4 | v5) & 0x7FFFFFFFFFFFFFFFLL) == 0;
  }
  return result;
}
