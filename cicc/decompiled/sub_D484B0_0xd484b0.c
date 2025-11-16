// Function: sub_D484B0
// Address: 0xd484b0
//
char __fastcall sub_D484B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  char result; // al

  v4 = (__int64 *)a2;
  v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a2 - 8);
    v4 = &v6[(unsigned __int64)v5 / 8];
  }
  else
  {
    v6 = (__int64 *)(a2 - v5);
  }
  v7 = v5 >> 5;
  v8 = v5 >> 7;
  if ( v8 )
  {
    v9 = &v6[16 * v8];
    while ( (unsigned __int8)sub_D48480(a1, *v6, a3, a4) )
    {
      if ( !(unsigned __int8)sub_D48480(a1, v6[4], v14, v15) )
      {
        v6 += 4;
        return v4 == v6;
      }
      if ( !(unsigned __int8)sub_D48480(a1, v6[8], v10, v11) )
        return v4 == v6 + 8;
      if ( !(unsigned __int8)sub_D48480(a1, v6[12], v12, v13) )
        return v4 == v6 + 12;
      v6 += 16;
      if ( v9 == v6 )
      {
        v7 = ((char *)v4 - (char *)v6) >> 5;
        goto LABEL_14;
      }
    }
    return v4 == v6;
  }
LABEL_14:
  if ( v7 != 2 )
  {
    if ( v7 != 3 )
    {
      if ( v7 != 1 )
        return 1;
      goto LABEL_22;
    }
    if ( !(unsigned __int8)sub_D48480(a1, *v6, a3, a4) )
      return v4 == v6;
    v6 += 4;
  }
  if ( !(unsigned __int8)sub_D48480(a1, *v6, a3, a4) )
    return v6 == v4;
  v6 += 4;
LABEL_22:
  result = sub_D48480(a1, *v6, a3, a4);
  if ( !result )
    return v6 == v4;
  return result;
}
