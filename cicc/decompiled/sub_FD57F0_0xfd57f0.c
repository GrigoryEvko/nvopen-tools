// Function: sub_FD57F0
// Address: 0xfd57f0
//
__int64 __fastcall sub_FD57F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  signed __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v10; // r13
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 4);
  v7 = a1;
  if ( v6 >> 2 > 0 )
  {
    v11 = a1 + 192 * (v6 >> 2);
    do
    {
      if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7, (__int64)(a3 + 1), 0) == 3 )
        return v7;
      v8 = v7 + 48;
      if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7 + 48, (__int64)(a3 + 1), 0) == 3 )
        return v8;
      v8 = v7 + 96;
      if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7 + 96, (__int64)(a3 + 1), 0) == 3 )
        return v8;
      v8 = v7 + 144;
      if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7 + 144, (__int64)(a3 + 1), 0) == 3 )
        return v8;
      v7 += 192;
    }
    while ( v11 != v7 );
    v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v7) >> 4);
  }
  if ( v6 == 2 )
  {
    v10 = (__int64)(a3 + 1);
    v8 = v7;
LABEL_22:
    if ( (unsigned __int8)sub_CF4D50(*a3, a4, v8, v10, 0) == 3 )
      return v8;
    v7 = v8 + 48;
    goto LABEL_18;
  }
  if ( v6 == 3 )
  {
    v10 = (__int64)(a3 + 1);
    v8 = v7;
    if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7, (__int64)(a3 + 1), 0) == 3 )
      return v8;
    v8 = v7 + 48;
    goto LABEL_22;
  }
  if ( v6 != 1 )
    return a2;
  v10 = (__int64)(a3 + 1);
LABEL_18:
  v8 = a2;
  if ( (unsigned __int8)sub_CF4D50(*a3, a4, v7, v10, 0) == 3 )
    return v7;
  return v8;
}
