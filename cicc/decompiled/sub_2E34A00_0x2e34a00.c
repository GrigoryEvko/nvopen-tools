// Function: sub_2E34A00
// Address: 0x2e34a00
//
signed __int64 __fastcall sub_2E34A00(__int64 *a1, char *a2, unsigned __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v4; // r13
  __int64 v6; // r15
  __int64 i; // r12
  __int64 v8; // r9
  __int64 v9; // r8

  result = a2 - (char *)a1;
  v4 = (__int64 *)a2;
  if ( a2 - (char *)a1 > 16 )
  {
    v6 = result >> 4;
    for ( i = ((result >> 4) - 2) / 2; ; --i )
    {
      result = (signed __int64)sub_2E301C0((__int64)a1, i, v6, a1[2 * i], a1[2 * i + 1]);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    do
    {
      while ( 1 )
      {
        result = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3;
        if ( (unsigned int)result < (*(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a1 >> 1) & 3) )
          break;
        v4 += 2;
        if ( a3 <= (unsigned __int64)v4 )
          return result;
      }
      v8 = *v4;
      *v4 = *a1;
      v9 = v4[1];
      v4 += 2;
      *(v4 - 1) = a1[1];
      result = (signed __int64)sub_2E301C0((__int64)a1, 0, (a2 - (char *)a1) >> 4, v8, v9);
    }
    while ( a3 > (unsigned __int64)v4 );
  }
  return result;
}
