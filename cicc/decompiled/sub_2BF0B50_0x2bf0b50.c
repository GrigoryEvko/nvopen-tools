// Function: sub_2BF0B50
// Address: 0x2bf0b50
//
__int64 __fastcall sub_2BF0B50(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 v3; // rcx
  _QWORD *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rcx
  _QWORD *v7; // rcx
  __int64 result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  v3 = 8LL * *(unsigned int *)(a1 + 24);
  v4 = &v2[(unsigned __int64)v3 / 8];
  v5 = v3 >> 3;
  v6 = v3 >> 5;
  if ( !v6 )
  {
LABEL_10:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return *v4;
LABEL_18:
        result = *v2;
        if ( a2 != *(_QWORD *)(*v2 + 128LL) )
          return *v4;
        return result;
      }
      result = *v2;
      if ( a2 == *(_QWORD *)(*v2 + 128LL) )
        return result;
      ++v2;
    }
    result = *v2;
    if ( a2 == *(_QWORD *)(*v2 + 128LL) )
      return result;
    ++v2;
    goto LABEL_18;
  }
  v7 = &v2[4 * v6];
  while ( 1 )
  {
    result = *v2;
    if ( a2 == *(_QWORD *)(*v2 + 128LL) )
      return result;
    result = v2[1];
    if ( a2 == *(_QWORD *)(result + 128) )
      return result;
    result = v2[2];
    if ( a2 == *(_QWORD *)(result + 128) )
      return result;
    result = v2[3];
    if ( a2 == *(_QWORD *)(result + 128) )
      return result;
    v2 += 4;
    if ( v7 == v2 )
    {
      v5 = v4 - v2;
      goto LABEL_10;
    }
  }
}
